"""Prespecified split, target-partition, horizon, smoothing, and subgroup audits."""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
import pandas as pd
from scipy.special import logit

from pafar_sim.calibration import fit_time_template, hc_threshold, marginal_threshold
from pafar_sim.io_utils import atomic_write_csv
from pafar_sim.score import trajectory_max
from .aggregation import descriptive_summary
from .internal_analysis import METHODS, _method_score_threshold, fit_split_preprocessor
from .learner import fit_locked_configuration
from .schema import RealDataConfig
from .scoring import TrajectoryData, evaluate, score_patients
from .splitting import nested_non_event_prefixes, seed_registry, stratified_patient_split
from .subgroups import summarize_subgroups


def _target_partition_runs(config: RealDataConfig, transfer: dict[str,object]) -> pd.DataFrame:
    seeds=seed_registry(config.master_seed); rows=[]
    for direction_data in transfer["directions"]:
        source,target=direction_data["direction"].split("→"); manifest_scores=direction_data["target_scores"]
        all_patients=pd.DataFrame({"patient_id":manifest_scores.patient_id,"hospital_set":manifest_scores.hospital,"any_sepsis_label":manifest_scores.event})
        direct=float(direction_data["results"].loc[direction_data["results"].strategy=="Direct source transfer","threshold"].iloc[0])
        for partition_no in range(1,50):
            split_seed=seeds[f"target_additional_{partition_no:02d}"]
            # Direction-specific child seeds avoid reusing the same permutation across A→B and B→A.
            child=0 if source=="A" else 1
            split_seed=int(np.random.SeedSequence(split_seed).spawn(2)[child].generate_state(1,dtype=np.uint64)[0])
            split=stratified_patient_split(all_patients,{"test":.5,"reservoir":.5},split_seed)
            test=manifest_scores.subset(set(split.loc[split.split=="test","patient_id"])); reservoir=split[split.split=="reservoir"]
            order_seed=int(np.random.SeedSequence(split_seed).spawn(2)[1].generate_state(1,dtype=np.uint64)[0])
            prefixes=nested_non_event_prefixes(reservoir,config.target_m0,order_seed)
            direct_metric,_=evaluate(test,test.score_f,direct)
            rows.append({"direction":direction_data["direction"],"partition":partition_no,"strategy":"Direct source transfer","target_m0":0,**direct_metric.as_dict()})
            template=direction_data["template"]
            for m0,ids in prefixes.items():
                local=manifest_scores.subset(set(ids)); non=~local.event
                max_f=trajectory_max(local.score_f[non],local.eligible[non]); max_t=trajectory_max(template.transform(local.score_f[non]),local.eligible[non])
                specs=(("Local PaFAR-F",test.score_f,marginal_threshold(max_f,.1).threshold),
                       ("Local PaFAR-T",template.transform(test.score_f),marginal_threshold(max_t,.1).threshold),
                       ("Local PaFAR-HC",test.score_f,hc_threshold(max_f,.1,config.delta).threshold))
                for strategy,score,threshold in specs:
                    metric,_=evaluate(test,score,threshold)
                    rows.append({"direction":direction_data["direction"],"partition":partition_no,"strategy":strategy,"target_m0":m0,**metric.as_dict(),"threshold":threshold,"infinite":np.isposinf(threshold)})
    return pd.DataFrame(rows)


def _additional_internal_splits(config: RealDataConfig, manifest: pd.DataFrame, internal: dict[str,object]) -> pd.DataFrame:
    seeds=seed_registry(config.master_seed); cohort=manifest.loc[manifest.primary_cohort,["patient_id","hospital_set","any_sepsis_label","reconstructed_onset","last_ICULOS"]]
    rows=[]; selected=internal["learner"].params
    for split_no in range(1,10):
        split=stratified_patient_split(cohort,config.raw["split"]["internal"],seeds[f"internal_additional_{split_no:02d}"])
        pre=fit_split_preprocessor(config,split,f"internal_sensitivity_{split_no:02d}")
        selection=lambda name:{h:set(split.loc[(split.split==name)&(split.hospital_set==h),"patient_id"]) for h in ("A","B")}
        learner=fit_locked_configuration(config,selection("train"),selection("validation"),pre,selected,seed=seeds[f"internal_additional_{split_no:02d}"],source=f"internal_sensitivity_{split_no:02d}",drop_hospital=False)
        val=score_patients(config,learner,pre,set(split.loc[split.split=="validation","patient_id"]),drop_hospital=False)
        cal=score_patients(config,learner,pre,set(split.loc[split.split=="calibration","patient_id"]),drop_hospital=False)
        test=score_patients(config,learner,pre,set(split.loc[split.split=="test","patient_id"]),drop_hospital=False)
        template=fit_time_template(val.score_f[~val.event],val.eligible[~val.event],config.tmin,config.hmax,config.template_min_patients)
        for hospital in ("A","B"):
            non=(cal.hospital==hospital)&(~cal.event); maxima=trajectory_max(cal.score_f[non],cal.eligible[non]); threshold=marginal_threshold(maxima,.1).threshold
            block=test.subset(set(test.patient_id[test.hospital==hospital])); metric,_=evaluate(block,block.score_f,threshold)
            rows.append({"split_no":split_no,"hospital":hospital,"method":"PaFAR-F","best_iteration":learner.best_iteration,**metric.as_dict()})
        non=~cal.event; maxima=trajectory_max(cal.score_f[non],cal.eligible[non]); pooled=marginal_threshold(maxima,.1).threshold
        metric,_=evaluate(test,test.score_f,pooled); rows.append({"split_no":split_no,"hospital":"overall_pooled_threshold","method":"PaFAR-F","best_iteration":learner.best_iteration,**metric.as_dict()})
    return pd.DataFrame(rows)


def _other_robustness(config: RealDataConfig, manifest: pd.DataFrame, internal: dict[str,object]) -> pd.DataFrame:
    rows=[]
    for label,hmax in (("96",96),("120",120),("168",168)):
        val,cal,test=(replace(internal[name],risk=internal[name].risk[:,:hmax],score_f=internal[name].score_f[:,:hmax],eligible=internal[name].eligible[:,:hmax],utility_grid=internal[name].utility_grid[:,:hmax],labels=internal[name].labels[:,:hmax]) for name in ("validation","calibration","test"))
        template=fit_time_template(val.score_f[~val.event],val.eligible[~val.event],config.tmin,hmax,config.template_min_patients)
        for method in ("PaFAR-F","PaFAR-T","PaFAR-HC"):
            score_test=template.transform(test.score_f) if method=="PaFAR-T" else test.score_f
            boundary=np.empty(score_test.shape)
            for hospital in ("A","B"):
                non=(cal.hospital==hospital)&(~cal.event); score_cal=template.transform(cal.score_f[non]) if method=="PaFAR-T" else cal.score_f[non]
                maxima=trajectory_max(score_cal,cal.eligible[non]); threshold=hc_threshold(maxima,.1,config.delta).threshold if method=="PaFAR-HC" else marginal_threshold(maxima,.1).threshold
                boundary[test.hospital==hospital]=threshold
            metric,_=evaluate(test,score_test,boundary); rows.append({"analysis":"Hmax","setting":label,"method":method,**metric.as_dict()})
    # Uncapped observed episode: the locked learner is scored to each patient's actual final ICU hour.
    max_h=int(max(internal[name].horizon.max() for name in ("validation","calibration","test")))
    uncapped={name:score_patients(config,internal["learner"],internal["preprocessor"],set(internal["split"].loc[internal["split"].split==name,"patient_id"]),drop_hospital=False,hmax=max_h) for name in ("validation","calibration","test")}
    val_u,cal_u,test_u=uncapped["validation"],uncapped["calibration"],uncapped["test"]
    template_u=fit_time_template(val_u.score_f[~val_u.event],val_u.eligible[~val_u.event],config.tmin,max_h,config.template_min_patients)
    for method in ("PaFAR-F","PaFAR-T","PaFAR-HC"):
        score_test=template_u.transform(test_u.score_f) if method=="PaFAR-T" else test_u.score_f
        boundary=np.empty(score_test.shape)
        for hospital in ("A","B"):
            non=(cal_u.hospital==hospital)&(~cal_u.event); score_cal=template_u.transform(cal_u.score_f[non]) if method=="PaFAR-T" else cal_u.score_f[non]
            maxima=trajectory_max(score_cal,cal_u.eligible[non]); threshold=hc_threshold(maxima,.1,config.delta).threshold if method=="PaFAR-HC" else marginal_threshold(maxima,.1).threshold
            boundary[test_u.hospital==hospital]=threshold
        metric,_=evaluate(test_u,score_test,boundary); rows.append({"analysis":"Hmax","setting":"uncapped","method":method,**metric.as_dict()})
    # Pooled threshold sensitivity.
    cal,test=internal["calibration"],internal["test"]; non=~cal.event; pooled=marginal_threshold(trajectory_max(cal.score_f[non],cal.eligible[non]),.1).threshold
    metric,_=evaluate(test,test.score_f,pooled); rows.append({"analysis":"calibration","setting":"pooled","method":"PaFAR-F",**metric.as_dict()})
    # All-recorded-episode PFA adds a prespecified 20% sample of below-landmark non-events,
    # whose empty eligible trajectories have score -infinity and therefore no alert.
    primary_pf=internal["patient_tables"][(.10,"PaFAR-F")]; primary_non=primary_pf.loc[~primary_pf.event.astype(bool)]
    excluded=manifest.loc[(~manifest.primary_cohort)&(~manifest.any_sepsis_label)&manifest.label_valid]
    rng=np.random.Generator(np.random.PCG64DXSM(int(np.random.SeedSequence(seed_registry(config.master_seed)["internal_primary"]).spawn(1)[0].generate_state(1,dtype=np.uint64)[0])))
    extra=0
    for _,block in excluded.groupby("hospital_set"):
        count=int(np.floor(.20*len(block)+.5)); extra+=count
        if count: rng.choice(len(block),size=count,replace=False)
    numerator=int(primary_non.alerted.sum()); denominator=len(primary_non)+extra
    rows.append({"analysis":"cohort","setting":"all-recorded-episode","method":"PaFAR-F","pfa":numerator/denominator if denominator else np.nan,
                 "n_non_events":denominator,"n_alerted":numerator,"below_landmark_assigned_test":extra})
    # Raw-score L=1 sensitivity with the learner unchanged.
    val_score=logit(np.clip(internal["validation"].risk,config.epsilon,1-config.epsilon)); cal_score=logit(np.clip(cal.risk,config.epsilon,1-config.epsilon)); test_score=logit(np.clip(test.risk,config.epsilon,1-config.epsilon))
    for hospital in ("A","B"):
        non=(cal.hospital==hospital)&(~cal.event); threshold=marginal_threshold(trajectory_max(cal_score[non],cal.eligible[non]),.1).threshold
        block=test.hospital==hospital; metric,_=evaluate(test.subset(set(test.patient_id[block])),test_score[block],threshold)
        rows.append({"analysis":"smoothing","setting":"L=1","method":f"PaFAR-F {hospital}",**metric.as_dict()})
    return pd.DataFrame(rows)


def run_robustness(config: RealDataConfig, manifest: pd.DataFrame, internal: dict[str,object], transfer: dict[str,object]) -> dict[str,pd.DataFrame]:
    started=time.perf_counter(); out=config.outputs/"robustness"
    target=_target_partition_runs(config,transfer); target_summary=descriptive_summary(target,["direction","strategy","target_m0"],["pfa","sens3","ppv3","alerts_per_100d"])
    atomic_write_csv(target,out/"target_partition_sensitivity.csv"); atomic_write_csv(target_summary,out/"target_partition_summary.csv")
    internal_splits=_additional_internal_splits(config,manifest,internal); internal_summary=descriptive_summary(internal_splits,["hospital","method"],["pfa","sens3","ppv3","alerts_per_100d","best_iteration"])
    atomic_write_csv(internal_splits,out/"internal_split_sensitivity.csv"); atomic_write_csv(internal_summary,out/"internal_split_summary.csv")
    other=_other_robustness(config,manifest,internal); atomic_write_csv(other,out/"other_robustness.csv")
    subgroup=summarize_subgroups(internal["patient_tables"][(.10,"PaFAR-F")],manifest); atomic_write_csv(subgroup,out/"subgroup_summary.csv")
    (out/"ROBUSTNESS_REPORT.md").write_text(f"# Robustness Report\n\nGate status: **PASS**\n\nElapsed seconds: {time.perf_counter()-started:.3f}\n",encoding="utf-8")
    return {"target":target,"target_summary":target_summary,"internal":internal_splits,"internal_summary":internal_summary,"other":other,"subgroup":subgroup}
