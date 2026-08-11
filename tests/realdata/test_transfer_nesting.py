import pandas as pd
from pafar_sim.realdata.splitting import nested_non_event_prefixes
def test_nested_prefixes_and_no_events():
    frame=pd.DataFrame({"patient_id":[f"p{i}" for i in range(1200)]+[f"e{i}" for i in range(10)],"any_sepsis_label":[False]*1200+[True]*10}); p=nested_non_event_prefixes(frame,[100,250,500,1000],8); assert set(p[100])<set(p[250])<set(p[500])<set(p[1000]); assert not any(x.startswith("e") for x in p[1000])

