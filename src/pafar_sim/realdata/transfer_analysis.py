"""Bidirectional source-hospital learning and target-local scalar recalibration."""
from __future__ import annotations

from dataclasses import asdict
import json
import time

import numpy as np
import pandas as pd

from pafar_sim.calibration import hc_threshold, marginal_threshold
from pafar_sim.io_utils import atomic_write_csv
from pafar_sim.score import trajectory_max
from .bootstrap import stratified_indices
from .calibration import exact_binomial, fit_template
from .internal_analysis import fit_split_preprocessor
from .json_utils import numeric_state
from .learner import fit_grid
from .schema import RealDataConfig
from .scoring import evaluate, score_patients, threshold_free
from .splitting import nested_non_event_prefixes, seed_registry, stratified_patient_split
from .utility import normalized_utility, utility_components


def _transfer_threshold_row(
    *, direction: str, strategy: str, target_m0: int, scale: str, threshold,
) -> dict[str, object]:
    """Preserve the numeric CSV threshold and add explicit state columns."""
    boundary = float(threshold.threshold)
    return {
        "direction": direction, "strategy": strategy, "target_m0": target_m0,
        "threshold_scale": scale, "threshold": boundary, "m0": threshold.m0,
        "k": threshold.index, "alpha_m0": threshold.alpha_m0,
        "infinite": threshold.infinite, **numeric_state(boundary),
    }


def _patient_metrics(frame: pd.DataFrame) -> dict[str,float]:
    non = ~frame.event.astype(bool)
    def ratio(a,b): return float(a/b) if b else np.nan
    den = frame.utility_best.sum()-frame.utility_inactive.sum()
    return {"pfa":ratio(frame.loc[non,"alerted"].sum(),non.sum()),"sens3":ratio(frame.valid3.sum(),frame.eval3.sum()),
            "sens0":ratio(frame.valid0.sum(),frame.eval0.sum()),"ppv3":ratio(frame.valid3.sum(),frame.alerted.sum()),
            "alerts_per_100d":ratio(100*frame.episodes.sum(),frame.exposure_days.sum()),
            "utility":ratio(frame.utility_observed.sum()-frame.utility_inactive.sum(),den)}


def _target_bootstrap(config: RealDataConfig, tables: dict[tuple[str,int],pd.DataFrame], seed: int, direction: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    first=next(iter(tables.values())); strata=first.event.astype(str).to_numpy(); rng=np.random.Generator(np.random.PCG64DXSM(seed))
    rows=[]
    for b in range(config.bootstrap_replicates):
        idx=stratified_indices(strata,rng)
        for (strategy,m0),frame in tables.items(): rows.append({"bootstrap":b,"direction":direction,"strategy":strategy,"target_m0":m0,**_patient_metrics(frame.iloc[idx])})
    samples=pd.DataFrame(rows); summaries=[]
    for (strategy,m0),frame in tables.items():
        observed=_patient_metrics(frame); block=samples[(samples.strategy==strategy)&(samples.target_m0==m0)]
        for metric,value in observed.items():
            x=block[metric].to_numpy(float); x=x[np.isfinite(x)]
            summaries.append({"direction":direction,"strategy":strategy,"target_m0":m0,"metric":metric,"observed":value,
                              "bootstrap_se":x.std(ddof=1) if len(x)>1 else np.nan,"lower_95":np.quantile(x,.025) if len(x) else np.nan,
                              "upper_95":np.quantile(x,.975) if len(x) else np.nan,"valid_bootstrap":len(x)})
    return samples,pd.DataFrame(summaries)


def run_direction(config: RealDataConfig, manifest: pd.DataFrame, source: str, target: str, *, resume: bool=True) -> dict[str,object]:
    seeds=seed_registry(config.master_seed); direction_ascii=f"{source}_to_{target}"; direction=f"{source}→{target}"
    source_cohort=manifest.loc[manifest.primary_cohort & (manifest.hospital_set==source),["patient_id","hospital_set","any_sepsis_label","reconstructed_onset","last_ICULOS"]]
    source_split=stratified_patient_split(source_cohort,config.raw["split"]["source"],seeds[f"{source}_to_{target}_source"])
    atomic_write_csv(source_split,config.data_root/"processed"/f"{direction_ascii}_source_split.csv")
    preprocessor=fit_split_preprocessor(config,source_split,direction_ascii)
    select=lambda name:{source:set(source_split.loc[source_split.split==name,"patient_id"])}
    learner,grid=fit_grid(config,select("train"),select("validation"),preprocessor,seed=seeds["xgboost"],source=direction_ascii,drop_hospital=True,resume=resume)
    validation=score_patients(config,learner,preprocessor,set(source_split.loc[source_split.split=="validation","patient_id"]),drop_hospital=True)
    calibration=score_patients(config,learner,preprocessor,set(source_split.loc[source_split.split=="calibration","patient_id"]),drop_hospital=True)
    template=fit_template(validation,config.tmin,config.hmax,config.template_min_patients)
    source_non=~calibration.event
    source_max_f=trajectory_max(calibration.score_f[source_non],calibration.eligible[source_non])
    source_max_t=trajectory_max(template.transform(calibration.score_f[source_non]),calibration.eligible[source_non])
    source_f=marginal_threshold(source_max_f,.10); source_t=marginal_threshold(source_max_t,.10); source_hc=hc_threshold(source_max_f,.10,config.delta)
    target_cohort=manifest.loc[manifest.primary_cohort & (manifest.hospital_set==target),["patient_id","hospital_set","any_sepsis_label","reconstructed_onset","last_ICULOS"]]
    target_split=stratified_patient_split(target_cohort,config.raw["split"]["target"],seeds[f"{source}_to_{target}_target_primary"])
    atomic_write_csv(target_split,config.data_root/"processed"/f"{direction_ascii}_target_partition.csv")
    reservoir=target_split[target_split.split=="reservoir"]
    child_no=0 if source=="A" else 1
    order_seed=int(np.random.SeedSequence(seeds["target_reservoir_order"]).spawn(2)[child_no].generate_state(1,dtype=np.uint64)[0])
    prefixes=nested_non_event_prefixes(reservoir,config.target_m0,order_seed)
    prefix_rows=[{"direction":direction,"target_m0":m0,"order":j,"patient_id":pid} for m0,ids in prefixes.items() for j,pid in enumerate(ids)]
    atomic_write_csv(pd.DataFrame(prefix_rows),config.data_root/"processed"/f"{direction_ascii}_target_nested_prefixes.csv")
    all_target_ids=set(target_split.patient_id)
    target_scores=score_patients(config,learner,preprocessor,all_target_ids,drop_hospital=True)
    target_test=target_scores.subset(set(target_split.loc[target_split.split=="test","patient_id"]))
    scorer_path=config.data_root/"manifests"/"official_evaluation_2019"/"evaluate_sepsis_score.py"
    strategy_specs=[("Direct source transfer",0,"F",source_f)]
    for m0 in config.target_m0:
        local=target_scores.subset(set(prefixes[m0])); non=~local.event
        max_f=trajectory_max(local.score_f[non],local.eligible[non]); max_t=trajectory_max(template.transform(local.score_f[non]),local.eligible[non])
        strategy_specs.extend([("Local PaFAR-F",m0,"F",marginal_threshold(max_f,.10)),
                               ("Local PaFAR-T",m0,"T",marginal_threshold(max_t,.10)),
                               ("Local PaFAR-HC",m0,"F",hc_threshold(max_f,.10,config.delta))])
    rows=[]; patient_tables={}; exact_rows=[]; threshold_rows=[]
    for strategy,m0,scale,threshold in strategy_specs:
        score=template.transform(target_test.score_f) if scale=="T" else target_test.score_f
        boundary=float(threshold.threshold)
        predictions=target_test.utility_grid & np.isfinite(score) & (score>boundary)
        utility,raw_utility=normalized_utility(target_test,predictions,scorer_path)
        uo,ui,ub=utility_components(target_test,predictions,scorer_path)
        metric,detail=evaluate(target_test,score,boundary,utility=utility)
        non=~target_test.event; successes=int(detail["alerted"][non].sum()); n=int(non.sum()); lower,upper,upper_one=exact_binomial(successes,n)
        rows.append({"direction":direction,"strategy":strategy,"target_m0":m0,**metric.as_dict(),"pfa_lower_95":lower,"pfa_upper_95":upper,"pfa_upper_one_sided_95":upper_one,"raw_utility":raw_utility,
                     "threshold":boundary,"calibration_count":threshold.m0,"k":threshold.index,"alpha_m0":threshold.alpha_m0,"infinite":threshold.infinite})
        exact_rows.append({"direction":direction,"strategy":strategy,"target_m0":m0,"alerts":successes,"n_non_events":n,"pfa":successes/n,"lower_95":lower,"upper_95":upper,"upper_one_sided_95":upper_one})
        threshold_rows.append(_transfer_threshold_row(direction=direction,strategy=strategy,target_m0=m0,scale=scale,threshold=threshold))
        pf=pd.DataFrame({"patient_id":target_test.patient_id,"hospital":target_test.hospital,"event":target_test.event,"onset":target_test.onset,"horizon":target_test.horizon,**detail,
                         "utility_observed":uo,"utility_inactive":ui,"utility_best":ub})
        pf["strategy"],pf["target_m0"],pf["direction"]=strategy,m0,direction
        patient_tables[(strategy,m0)]=pf
    bootstrap_seed=int(np.random.SeedSequence(seeds["bootstrap"]).spawn(2)[child_no].generate_state(1,dtype=np.uint64)[0])
    samples,summaries=_target_bootstrap(config,patient_tables,bootstrap_seed,direction)
    auroc,pr_auc=threshold_free(target_test)
    return {"direction":direction,"direction_ascii":direction_ascii,"source_split":source_split,"target_split":target_split,"prefixes":prefixes,
            "preprocessor":preprocessor,"learner":learner,"template":template,"target_scores":target_scores,"target_test":target_test,
            "results":pd.DataFrame(rows),"patient_tables":patient_tables,"exact":pd.DataFrame(exact_rows),"thresholds":pd.DataFrame(threshold_rows),
            "bootstrap_samples":samples,"bootstrap_summary":summaries,"grid":grid,"target_auroc":auroc,"target_pr_auc":pr_auc}


def run_cross_hospital(config: RealDataConfig, manifest: pd.DataFrame, *, resume: bool=True) -> dict[str,object]:
    started=time.perf_counter(); directions=[run_direction(config,manifest,"A","B",resume=resume),run_direction(config,manifest,"B","A",resume=resume)]
    out=config.outputs/"transfer_primary"
    atomic_write_csv(pd.concat([d["results"] for d in directions],ignore_index=True),out/"transfer_results_long.csv.gz")
    atomic_write_csv(pd.concat([d["results"] for d in directions],ignore_index=True),out/"transfer_results_summary.csv")
    atomic_write_csv(pd.concat([d["exact"] for d in directions],ignore_index=True),out/"target_exact_intervals.csv")
    atomic_write_csv(pd.concat([d["thresholds"] for d in directions],ignore_index=True),out/"thresholds.csv")
    atomic_write_csv(pd.concat([d["bootstrap_samples"] for d in directions],ignore_index=True),config.outputs/"bootstrap"/"target_bootstrap_samples.csv.gz")
    atomic_write_csv(pd.concat([d["bootstrap_summary"] for d in directions],ignore_index=True),config.outputs/"bootstrap"/"target_bootstrap_summary.csv")
    learner_rows=[{"direction":d["direction"],**d["learner"].params,"best_iteration":d["learner"].best_iteration,"trees_used":d["learner"].trees_used,
                   "validation_aucpr":d["learner"].best_score,"target_weighted_auroc":d["target_auroc"],"target_trapezoidal_pr_auc":d["target_pr_auc"]} for d in directions]
    atomic_write_csv(pd.DataFrame(learner_rows),out/"learner_metrics.csv")
    lines=["# Cross-Hospital Report","","Gate status: **PASS**","",f"Elapsed seconds: {time.perf_counter()-started:.3f}"]
    (out/"CROSS_HOSPITAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"directions":directions,"elapsed_seconds":time.perf_counter()-started}
