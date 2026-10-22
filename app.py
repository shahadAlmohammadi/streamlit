"""BALMIRSAD | bilingual, theme-aware, local-first Streamlit command center.
All data is synthetic. Optional Ollama inference runs on localhost; no cloud API used.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import html
import json
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BALMIRSAD | بالمرصاد",page_icon="✦",layout="wide",initial_sidebar_state="expanded")

WORDS = {
"ar": {
"brand":"بالمِرصاد","tagline":"منظومة الرصد البيئي الذكي","demo":"بيانات تجريبية · المدينة المنورة","online":"مركز العمليات نشط","nav":"مساحات العمل","command":"مركز القيادة","explore":"مستكشف البلاغات","analytics":"استوديو التحليلات","lab":"مختبر القرارات","assistant":"مساعد المرصاد AI","scope":"فلاتر البيانات","districts":"الأحياء","severity":"درجة الخطورة","status":"حالة البلاغ","period":"الفترة الزمنية","days":"يوم","all":"الكل","reset":"إعادة ضبط الديمو","records":"بلاغات مرصودة","critical":"بلاغات حرجة مفتوحة","closed":"نسبة الإغلاق","response":"متوسط الاستجابة","hours":"ساعة","since":"خلال الفترة المحددة","priority":"الأولوية","map":"مراقبة التوزيع المكاني","map_sub":"خريطة إحداثيات تفاعلية دون خرائط خارجية","alerts":"غرفة الإنذار المبكر","alerts_sub":"البلاغات الأعلى أولوية","risk_dist":"توزيع مستوى الخطورة","top_districts":"الأحياء الأعلى رصدًا","latest":"آخر البلاغات","command_title":"ذكاء المدينة، في شاشة واحدة.","command_sub":"من الرصد إلى التنبّه واتخاذ القرار. منصة تشغيلية تفاعلية للتحليلات، وإدارة البلاغات، ومحاكاة الاستجابة.","explore_title":"كل موقع له تفاصيل تستحق الاكتشاف.","explore_sub":"افهم أسباب الخطورة، راجع موقع البلاغ، وحدث فريق الاستجابة وحالة المعالجة.","analytics_title":"حوّل البيانات إلى أثر واضح.","analytics_sub":"تحليلات تشغيلية تفاعلية تكشف التوجهات والأنماط والفجوات.","lab_title":"ماذا لو غيّرنا القرار؟","lab_sub":"عدّل أوزان الأولوية وطاقة الفرق وشاهد تأثير خياراتك فورًا.","ai_title":"اسأل بياناتك. احصل على تفسير.","ai_sub":"مساعد سياقي يعرف الفلاتر والبلاغات الحالية؛ يدعم نموذج Ollama المحلي اختياريًا.","choose":"اختر بلاغًا","details":"بطاقة البلاغ","factors":"عوامل تحديد الأولوية","change":"تحديث البلاغ","team":"الفريق","save":"حفظ التحديث","saved":"تم تحديث البلاغ في جلسة العرض","trend":"حركة البلاغات يوميًا","cycle":"توزيع الحالات","heat":"الحي × نوع المخلفات","resp_district":"الاستجابة حسب الحي","export":"تصدير البيانات","export_csv":"تنزيل CSV","size":"تأثير مساحة المخلفات","population":"تأثير الكثافة السكانية","facility":"تأثير القرب من المرافق","open":"بلاغات تحتاج تدخلًا","above80":"أولوية أعلى من 80","maxscore":"أعلى درجة أولوية","ranking":"الترتيب الديناميكي","dispatch":"محاكي توزيع الفرق","teams":"عدد الفرق المتاحة","capacity":"طاقة كل فريق اليومية","backlog":"المتبقي بعد يوم افتراضي","dispatch_btn":"توليد خطة التوزيع","dispatch_done":"تم توليد خطة افتراضية دون تعديل البيانات","export_plan":"تصدير الخطة","formula":"كيف تعمل الأولوية؟","formula_desc":"متوسط مرجّح للمساحة والكثافة والقرب من المرافق مع تعديل للخطورة. هذه قواعد تفسيرية توضيحية وليست نموذج تعلم آلي مدرب.","source":"مصدر البيانات: محاكاة تعليمية، وليست بيانات بلدية حقيقية","empty":"لا توجد بيانات تطابق الفلاتر الحالية.","select_model":"وضع المساعد","insights":"تحليل فوري","local_ai":"ذكاء اصطناعي محلي (Ollama)","model":"اسم النموذج المحلي","connection":"حالة الاتصال","connected":"Ollama متاح","disconnected":"Ollama غير متاح؛ يمكنك استخدام التحليل الفوري","chat_placeholder":"مثال: ما أكثر الأحياء بلاغات؟ وكيف نقلل الأولويات الحرجة؟","send":"إرسال","questions":"أسئلة مقترحة","q1":"لخص حالة البلاغات الحالية","q2":"ما أكثر الأحياء بلاغات؟","q3":"ما أبرز المخاطر المفتوحة؟","q4":"اقترح خطة عمل للفرق","context":"المساعد يستخدم البيانات المفلترة المعروضة حاليًا","clear_chat":"مسح المحادثة","no_cloud":"بدون إرسال البيانات لخدمات سحابية في وضع Ollama المحلي","model_note":"يتطلب تثبيت Ollama وتنزيل النموذج مسبقًا. لا يوجد نموذج لغوي مدمج داخل هذا الملف.","simulated":"التحليل الفوري مبني على حسابات وقواعد، وليس نموذجًا لغويًا.","ai_failed":"تعذّر التواصل مع النموذج المحلي. تأكد من تشغيل Ollama وتحميل النموذج.","you":"أنت","bot":"مساعد المرصاد","new":"جديد","progress":"قيد المعالجة","done":"مغلق","low":"منخفض","medium":"متوسط","high":"مرتفع","urgent":"حرج","none":"غير مسند","north":"فريق الشمال","south":"فريق الجنوب","east":"فريق الشرق","west":"فريق الغرب","construction":"مخلفات بناء","solid":"نفايات صلبة","illegal":"تجمع عشوائي","commercial":"مخلفات تجارية","green":"مخلفات خضراء","id":"رقم البلاغ","district":"الحي","type":"النوع","date":"تاريخ الرصد","area":"المساحة م²","score":"درجة الأولوية","confidence":"ثقة الرصد","lat":"خط العرض","lon":"خط الطول","download":"تنزيل","coverage":"التغطية التشغيلية","pending":"بلاغات متبقية","site_count":"عدد البلاغات","total_team":"السعة اليومية","days_setting":"آخر","help":"ملاحظة: المساعد لا يُجري تغييرات فعلية على أنظمة خارجية."},
"en": {
"brand":"BALMIRSAD","tagline":"Smart Environmental Intelligence","demo":"Synthetic data · Madinah","online":"Operations center live","nav":"WORKSPACES","command":"Command Center","explore":"Incident Explorer","analytics":"Analytics Studio","lab":"Decision Lab","assistant":"Mirsad AI Copilot","scope":"DATA FILTERS","districts":"Districts","severity":"Severity","status":"Incident status","period":"Time window","days":"days","all":"All","reset":"Reset demo","records":"Detected incidents","critical":"Open critical incidents","closed":"Resolution rate","response":"Avg. response time","hours":"hrs","since":"Within selected range","priority":"Priority","map":"Spatial monitoring","map_sub":"Interactive coordinate map, no external map tiles","alerts":"Early warning center","alerts_sub":"Highest-priority open incidents","risk_dist":"Severity mix","top_districts":"Most affected districts","latest":"Recent incidents","command_title":"A smarter city. One command center.","command_sub":"From detection to decisive action. Explore spatial intelligence, real-time filtering, and response simulation in one workspace.","explore_title":"Every location tells a story.","explore_sub":"Understand risk drivers, review each incident, and update response ownership and status.","analytics_title":"Turn signals into actionable insight.","analytics_sub":"Interactive operational analytics reveal patterns, trends, and gaps in response.","lab_title":"What if your next decision changed?","lab_sub":"Tune priority weights and team capacity to instantly simulate operational outcomes.","ai_title":"Ask your data. Get clear answers.","ai_sub":"A filter-aware copilot for current incident data, with optional local Ollama AI.","choose":"Select an incident","details":"Incident profile","factors":"Priority factors","change":"Update incident","team":"Assigned team","save":"Save changes","saved":"Demo session updated","trend":"Daily incident trend","cycle":"Resolution pipeline","heat":"District × waste category","resp_district":"Response by district","export":"Export center","export_csv":"Download CSV","size":"Waste area influence","population":"Population density influence","facility":"Public facility proximity","open":"Open incidents","above80":"Priority scores ≥ 80","maxscore":"Peak priority score","ranking":"Dynamic priority ranking","dispatch":"Response allocation simulator","teams":"Available crews","capacity":"Incidents per crew per day","backlog":"Estimated one-day backlog","dispatch_btn":"Generate response plan","dispatch_done":"Illustrative dispatch plan created, no data changed","export_plan":"Export plan","formula":"How is priority calculated?","formula_desc":"A weighted average of waste footprint, population, and facility proximity plus a severity adjustment. This is an explainable illustrative rule set, not a trained ML model.","source":"Data source: educational simulation, not real municipal data","empty":"No incidents match the current filters.","select_model":"Copilot mode","insights":"Instant insights","local_ai":"Local AI (Ollama)","model":"Local model name","connection":"Connection","connected":"Ollama is available","disconnected":"Ollama is not available; use instant insights instead","chat_placeholder":"Ask: Which district needs attention? Suggest an action plan.","send":"Send","questions":"Suggested questions","q1":"Summarize the current incidents","q2":"Which districts have the most incidents?","q3":"What are the top open risks?","q4":"Suggest an allocation plan","context":"The copilot understands currently filtered records","clear_chat":"Clear conversation","no_cloud":"Local Ollama mode sends no data to cloud services","model_note":"Install Ollama and pull a model first. No language model is bundled with this application.","simulated":"Instant insights are rule-based calculations, not generative AI.","ai_failed":"Could not reach the local model. Check Ollama and the model installation.","you":"You","bot":"Mirsad Copilot","new":"New","progress":"In progress","done":"Resolved","low":"Low","medium":"Moderate","high":"High","urgent":"Critical","none":"Unassigned","north":"North crew","south":"South crew","east":"East crew","west":"West crew","construction":"Construction debris","solid":"Solid waste","illegal":"Illegal dumping","commercial":"Commercial waste","green":"Green waste","id":"Incident ID","district":"District","type":"Category","date":"Detected at","area":"Footprint m²","score":"Priority score","confidence":"Detection confidence","lat":"Latitude","lon":"Longitude","download":"Download","coverage":"Operations coverage","pending":"Remaining incidents","site_count":"Incident count","total_team":"Daily capacity","days_setting":"Last","help":"Copilot does not act on external production systems."}
}

DISTRICTS = [ ("العوالي","Al Awali",24.449,39.636),("قباء","Quba",24.432,39.607),("العزيزية","Al Aziziyah",24.479,39.534),("قربان","Qurban",24.456,39.626),("شوران","Shuran",24.404,39.607),("الخالدية","Al Khalidiyah",24.471,39.660),("العريض","Al Aridh",24.487,39.655),("العيون","Al Uyun",24.508,39.583),("بني حارثة","Bani Harithah",24.499,39.634),("الدفاع","Al Difa",24.420,39.562),("السلام","Al Salam",24.464,39.567),("الملك فهد","King Fahd",24.518,39.660)]
WASTE=[("مخلفات بناء","Construction debris"),("نفايات صلبة","Solid waste"),("تجمع عشوائي","Illegal dumping"),("مخلفات تجارية","Commercial waste"),("مخلفات خضراء","Green waste")]
SEVERITY=[("منخفض","Low"),("متوسط","Moderate"),("مرتفع","High"),("حرج","Critical")]
STATUSES=[("جديد","New"),("قيد المعالجة","In progress"),("مغلق","Resolved")]
TEAMS=[("غير مسند","Unassigned"),("فريق الشمال","North crew"),("فريق الجنوب","South crew"),("فريق الشرق","East crew"),("فريق الغرب","West crew")]

@st.cache_data(show_spinner=False)
def make_demo(day):
    rng=np.random.default_rng(20261022)
    rows=[]
    for i in range(286):
        j=int(rng.choice(12,p=np.array([.135,.09,.075,.095,.115,.08,.085,.06,.07,.075,.06,.06])))
        d=DISTRICTS[j]; level=int(rng.choice(4,p=[.25,.35,.27,.13])); status=int(rng.choice(3,p=[.20,.29,.51])); age=int(rng.integers(0,90))
        observed=datetime.combine(day-timedelta(days=age),datetime.min.time())+timedelta(hours=int(rng.integers(6,22)),minutes=int(rng.integers(0,60)))
        rows.append(dict(id=f"MED-26-{i+801:04d}",district=j,kind=int(rng.integers(0,5)),severity=level,status=status,detected=observed,
            area=round(float(rng.lognormal(3.5,.65)),1),confidence=int(rng.integers(82,99)),population=int(rng.integers(15,100)),facility=int(rng.integers(0,100)),
            response=round(float(rng.uniform(1.5,12)+(level==3)*.8+(status==0)*4),1),lat=round(d[2]+float(rng.normal(0,.011)),5),lon=round(d[3]+float(rng.normal(0,.012)),5),team=int(rng.integers(0,5))))
    return pd.DataFrame(rows)

if "records" not in st.session_state: st.session_state.records=make_demo(date.today()).copy()
if "lang" not in st.session_state: st.session_state.lang="ar"
if "theme_mode" not in st.session_state: st.session_state.theme_mode="Dark"
if "dispatch" not in st.session_state: st.session_state.dispatch=[]
if "messages" not in st.session_state: st.session_state.messages=[]
lang=st.session_state.lang
W=WORDS[lang]
def t(key): return W[key]
def loc(pairs, code=None): return pairs[0 if (code or lang)=="ar" else 1]
def dname(v): return loc(DISTRICTS[int(v)])
def sname(v): return loc(SEVERITY[int(v)])
def stname(v): return loc(STATUSES[int(v)])
def kname(v): return loc(WASTE[int(v)])
def teamname(v): return loc(TEAMS[int(v)])

dark=st.session_state.theme_mode=="Dark"
P= dict(bg="#07151D",sidebar="#0B1C25",panel="#112630",panel2="#132D36",edge="#24424A",text="#ECFAF7",muted="#94B2B9",accent="#48E7BB",blue="#7FB6FF",gold="#FFC37A",red="#FF758B",grid="#25414A",hero="#12382F") if dark else dict(bg="#F1F6F5",sidebar="#FFFFFF",panel="#FFFFFF",panel2="#E7F5F0",edge="#D8E7E4",text="#162A35",muted="#637F88",accent="#087E68",blue="#3C72C5",gold="#B57626",red="#D84865",grid="#E0EAE9",hero="#E4F4ED")
RTL=lang=="ar"
css="""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[data-testid="stApp"] {font-family:'DM Sans','IBM Plex Sans Arabic',Tahoma,sans-serif;}
.stApp {background:radial-gradient(ellipse 52% 28% at 78% -10%, {{hero}},transparent),{{bg}};color:{{text}};direction:{{direction}};}
[data-testid="stSidebar"] {background:{{sidebar}} !important;border-{{side}}:1px solid {{edge}};}
[data-testid="stSidebar"] * {direction:{{direction}};}
[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.2rem;max-width:1620px;padding-bottom:3rem;}
h1,h2,h3,h4,p,label {color:{{text}};}
[data-testid="stMetric"] {background:{{panel}};border:1px solid {{edge}};border-radius:17px;padding:15px;}
[data-testid="stMetricLabel"], [data-testid="stWidgetLabel"] {color:{{muted}};}
[data-testid="stMetricValue"] {color:{{text}};}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {border:1px solid {{edge}};border-radius:17px;overflow:hidden;}
[data-testid="stChatMessage"] {background:{{panel}};border:1px solid {{edge}};border-radius:16px;}
[data-testid="stChatInput"] {direction:{{direction}};}
.stButton button[kind="primary"] {background:{{accent}};border:0;color:{{primaryText}};font-weight:800;}
.stButton button {border-radius:12px;border:1px solid {{edge}};}
.hero {background:linear-gradient(110deg,{{hero}},{{panel}});border:1px solid {{edge}};border-radius:25px;padding:30px;margin-bottom:20px;position:relative;overflow:hidden;box-shadow:0 12px 44px #001b1914;}
.hero:after{content:'✦';position:absolute;{{opposite}}:40px;top:-70px;font-size:210px;color:{{accent}};opacity:.075;pointer-events:none;}
.eyebrow{letter-spacing:2px;color:{{accent}};font-size:.72rem;font-weight:800;}
.hero h1{font-size:clamp(1.7rem,3vw,2.65rem);line-height:1.35;letter-spacing:-.7px;margin:9px 0;color:{{text}};font-weight:800;}
.hero p{font-size:1rem;line-height:1.85;color:{{muted}};margin:0;max-width:820px;}
.live{display:inline-block;background:{{panel2}};color:{{accent}};border:1px solid {{edge}};padding:7px 13px;border-radius:80px;font-size:.8rem;margin-top:15px;}
.card{background:{{panel}};border:1px solid {{edge}};border-radius:18px;padding:19px;margin-bottom:12px;box-shadow:0 8px 25px #082b2110;}
.card .label{font-size:.82rem;color:{{muted}};margin-bottom:9px;}
.card .number{font-size:2rem;color:{{text}};font-weight:800;line-height:1.3;}
.card .note{font-size:.8rem;color:{{muted}};margin-top:8px;}
.h{font-size:1.08rem;color:{{text}};font-weight:800;margin:17px 0 9px;}
.sub{color:{{muted}};font-size:.78rem;line-height:1.7;}
.brand{font-size:2rem;font-weight:800;color:{{text}};letter-spacing:-.5px;}
.footer{text-align:center;padding:36px 6px 0;font-size:.73rem;color:{{muted}};}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:{{edge}};}
</style>
"""
for k,v in dict(P,direction="rtl" if RTL else "ltr",side="left" if RTL else "right",opposite="left" if RTL else "right",primaryText="#06241E" if dark else "#FFFFFF").items(): css=css.replace("{{"+k+"}}",v)
st.markdown(css,unsafe_allow_html=True)

# Motion + layout polish. Honors OS reduced-motion preference.
extra_css = """
<style>
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { overflow-x: clip; }
.block-container { max-width: 1480px !important; padding-inline: clamp(1rem,2.5vw,2.5rem) !important; padding-bottom: 6rem !important; }
[data-testid="stHorizontalBlock"] { align-items: stretch; }
[data-testid="column"] { min-width: 0 !important; }
[data-testid="column"] > div { min-width: 0 !important; }
[data-testid="stPlotlyChart"] { min-width: 0; max-width: 100%; }
[data-testid="stDataFrame"] { max-width: 100%; }
[data-testid="stSidebar"] { min-width: 0; }
.stApp .card { min-width:0; overflow-wrap:anywhere; transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease; animation: fadeFloat .55s ease both; }
.stApp .card:hover { transform:translateY(-3px); border-color: {{accent}}; box-shadow: 0 15px 35px #00000016; }
.stApp .hero { animation: heroEnter .6s cubic-bezier(.2,.7,.3,1) both; }
.stApp .hero::after { animation: floatSymbol 9s ease-in-out infinite alternate; }
.stApp .live::before { content: '●'; display: inline-block; color:{{accent}}; margin-inline-end: 7px; animation: livePulse 2.2s ease-in-out infinite; }
.stApp .live { max-width:100%; white-space:normal; }
.stButton button { transition: transform .17s ease, box-shadow .17s ease, border-color .17s ease; }
.stButton button:hover {transform:translateY(-1px); border-color:{{accent}}; box-shadow: 0 8px 20px #00000014; }
[data-testid="stMetric"] { min-width:0; overflow:hidden; }
[data-testid="stMetricValue"] { overflow-wrap:anywhere; font-size:clamp(1.2rem,2vw,2.1rem) !important; }
/* The floating assistant is anchored to the viewport, not the dashboard grid. */
.st-key-mirsad_floating_chat { position:fixed !important; bottom:22px !important; left:22px !important; right:auto !important; z-index:999991 !important; width: auto !important; max-width: min(410px, calc(100vw - 24px)) !important; filter:drop-shadow(0 12px 22px #0005); }
.st-key-mirsad_floating_chat > div { width:auto !important; }
.st-key-mirsad_floating_chat [data-testid="stPopover"] > button { background:linear-gradient(120deg,{{accent}},{{blue}}) !important; color:{{primaryText}} !important; border:1px solid #ffffff44 !important; border-radius:100px !important; padding: .75rem 1.25rem !important; font-weight:800 !important; min-height:53px; box-shadow:0 10px 32px #00000035; animation:enterChat .8s cubic-bezier(.15,.85,.2,1) both; }
.st-key-mirsad_floating_chat [data-testid="stPopover"] > button:hover { transform:translateY(-4px) !important; }
[data-testid="stPopoverBody"] { max-height:min(74vh,720px) !important; overflow-y:auto !important; border-radius:20px !important; border:1px solid {{edge}} !important; background:{{panel}} !important; color:{{text}} !important; box-shadow:0 24px 80px #0005 !important; }
[data-testid="stPopoverBody"] [data-testid="stChatMessage"] { padding:.55rem .65rem; border-radius:14px; }
[data-testid="stPopoverBody"] .stMarkdown p {overflow-wrap:anywhere;}
[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] { gap:.55rem !important; }
[data-testid="stPopoverBody"] [data-testid="stTextInput"] input { border-radius:11px; }
@keyframes fadeFloat { from { opacity:0; transform:translateY(10px); } to {opacity:1;transform:translateY(0);} }
@keyframes heroEnter { from {opacity:0;transform:translateY(14px) scale(.991);} to{opacity:1;transform:translateY(0) scale(1);} }
@keyframes floatSymbol {from{transform:translateY(0) rotate(-9deg)} to{transform:translateY(28px) rotate(7deg)}}
@keyframes livePulse {0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.75)}}
@keyframes enterChat {from{opacity:0;transform:translateY(26px) scale(.94)}to{opacity:1;transform:translateY(0) scale(1)}}
@media(max-width:800px) { .block-container {padding-inline:.85rem !important;} .hero {padding:19px !important;} .hero h1 {font-size:1.65rem !important;} .st-key-mirsad_floating_chat{left:12px !important;bottom:12px !important;} }
@media(prefers-reduced-motion:reduce) { *,*::before,*::after {animation:none !important;transition:none !important;} }
</style>
"""
for k,v in dict(P,primaryText="#06241E" if dark else "#FFFFFF").items():extra_css=extra_css.replace("{{"+k+"}}",v)
st.markdown(extra_css,unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="eyebrow">SMART CITY INTELLIGENCE</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="brand">✦ {html.escape(t("brand"))}</div><div class="sub">{html.escape(t("tagline"))}</div>',unsafe_allow_html=True)
    st.divider()
    c1,c2=st.columns(2)
    with c1: selected_language=st.selectbox("Language / اللغة",["العربية","English"],index=0 if lang=="ar" else 1,key="language_picker")
    with c2: selected_theme=st.selectbox("Theme / المظهر",["Dark","Light"],index=0 if dark else 1,key="theme_picker")
    new_lang="ar" if selected_language=="العربية" else "en"
    if new_lang!=lang or selected_theme!=st.session_state.theme_mode:
        st.session_state.lang=new_lang; st.session_state.theme_mode=selected_theme; st.rerun()
    st.divider()
    st.caption(t("nav"))
    pages=["command","explore","analytics","lab"]
    icons=["◈","◎","▥","⚡"]
    chosen=st.radio("Navigation",pages,format_func=lambda x:icons[pages.index(x)]+"  "+t(x),label_visibility="collapsed",key="navigation")
    st.divider()
    st.caption(t("scope"))
    selected_areas=st.multiselect(t("districts"),list(range(12)),format_func=dname,placeholder=t("all"))
    selected_levels=st.multiselect(t("severity"),list(range(4)),format_func=sname,placeholder=t("all"))
    selected_states=st.multiselect(t("status"),list(range(3)),format_func=stname,placeholder=t("all"))
    days=st.select_slider(t("period"),options=[7,14,30,60,90],value=90,format_func=lambda x:f"{t('days_setting')} {x} {t('days')}")
    st.divider()
    if st.button("↻ "+t("reset"),use_container_width=True):
        st.session_state.records=make_demo(date.today()).copy();st.session_state.dispatch=[];st.session_state.messages=[];st.rerun()
    st.caption(t("source"))

all_records=st.session_state.records.copy()
f=all_records[all_records.detected.dt.date>=date.today()-timedelta(days=days-1)].copy()
if selected_areas:f=f[f.district.isin(selected_areas)]
if selected_levels:f=f[f.severity.isin(selected_levels)]
if selected_states:f=f[f.status.isin(selected_states)]


def priority(df,ws=35,wp=35,wf=30):
    out=df.copy()
    if out.empty:out["score"]=pd.Series(dtype=float);return out
    size=np.clip(out.area.to_numpy()/175,0,1)*100
    base=(ws*size+wp*out.population.to_numpy()+wf*out.facility.to_numpy())/max(ws+wp+wf,1)
    out["score"]=np.clip(base+out.severity.map({0:0,1:4,2:8,3:13}).to_numpy(),0,100).round(1)
    return out

def heading(kicker,title,sub):
    st.markdown(f'<div class="hero"><div class="eyebrow">{html.escape(kicker)} / BALMIRSAD</div><h1>{html.escape(title)}</h1><p>{html.escape(sub)}</p><span class="live">{html.escape(t("demo"))} · {date.today():%Y/%m/%d}</span></div>',unsafe_allow_html=True)

def section(label,note=""):
    st.markdown(f'<div class="h">{html.escape(label)}</div>',unsafe_allow_html=True)
    if note:st.markdown(f'<div class="sub">{html.escape(note)}</div>',unsafe_allow_html=True)

def tile(label,value,note,color=None):
    st.markdown(f'<div class="card"><div class="label">{html.escape(label)}</div><div class="number">{html.escape(str(value))}</div><div class="note" style="color:{color or P["muted"]}">{html.escape(note)}</div></div>',unsafe_allow_html=True)

def style(fig,h=335,legend=False):
    fig.update_layout(height=h,margin=dict(l=18,r=18,t=25,b=24),paper_bgcolor=P["panel"],plot_bgcolor=P["panel"],font=dict(family="Arial, Tahoma",color=P["text"],size=12),showlegend=legend,
        xaxis=dict(gridcolor=P["grid"],zeroline=False),yaxis=dict(gridcolor=P["grid"],zeroline=False),hoverlabel=dict(bgcolor=P["panel2"],font_color=P["text"]))
    return fig

RISK_COLORS={0:P["accent"],1:P["blue"],2:P["gold"],3:P["red"]}
STATE_COLORS={0:P["blue"],1:P["gold"],2:P["accent"]}

def scatter_map(df):
    fig=go.Figure()
    for n in range(4):
        p=df[df.severity==n]
        if not p.empty:
            fig.add_trace(go.Scatter(x=p.lon,y=p.lat,mode="markers",name=sname(n),marker=dict(size=np.clip(p.area**.4*2,8,24),color=RISK_COLORS[n],opacity=.88,line=dict(width=1,color=P["panel"])),customdata=np.stack([p.id,p.district.map(dname),p.kind.map(kname)],axis=-1),hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<extra></extra>"))
    for d in DISTRICTS:fig.add_annotation(x=d[3],y=d[2],text=loc(d),showarrow=False,font=dict(color=P["muted"],size=10),opacity=.78)
    fig.update_xaxes(title=t("lon"),tickformat=".3f");fig.update_yaxes(title=t("lat"),tickformat=".3f",scaleanchor="x",scaleratio=1)
    fig.update_layout(legend=dict(orientation="h",y=1.14,x=0),dragmode="pan")
    return style(fig,423,True)

def display_table(df,cols):
    temp=df.copy()
    for n,func in [("district",dname),("kind",kname),("severity",sname),("status",stname),("team",teamname)]:
        if n in temp:temp[n]=temp[n].map(func)
    labels={"id":t("id"),"district":t("district"),"kind":t("type"),"severity":t("severity"),"status":t("status"),"detected":t("date"),"area":t("area"),"score":t("score"),"team":t("team")}
    return temp[cols].rename(columns=labels)

def context_snapshot(df, lang_code):
    """Send aggregate counts and a small anonymized sample, not the entire record set."""
    pending=df[df.status!=2]
    ranked=priority(pending).sort_values("score",ascending=False)
    return {
        "dataset_type":"synthetic demonstration; not real city data",
        "filtered_count":len(df),"open_count":len(pending),"critical_open":int((pending.severity==3).sum()),
        "resolved_count":int((df.status==2).sum()),"mean_response_hours":round(float(df.response.mean()),2) if len(df) else 0,
        "district_counts":{loc(DISTRICTS[int(k)],lang_code):int(v) for k,v in df.district.value_counts().items()},
        "status_counts":{loc(STATUSES[int(k)],lang_code):int(v) for k,v in df.status.value_counts().items()},
        "severity_counts":{loc(SEVERITY[int(k)],lang_code):int(v) for k,v in df.severity.value_counts().items()},
        "waste_counts":{loc(WASTE[int(k)],lang_code):int(v) for k,v in df.kind.value_counts().items()},
        "top_open_incidents":[{"id":r.id,"district":loc(DISTRICTS[int(r.district)],lang_code),"severity":loc(SEVERITY[int(r.severity)],lang_code),"priority":float(r.score)} for _,r in ranked.head(8).iterrows()],
        "priority_method":"illustrative rules based on waste area, population density, proximity and severity; not trained AI"
    }


def instant_answer(question,df,lang_code):
    """Deterministic, explainable fallback. Not represented as generative AI."""
    data=context_snapshot(df,lang_code)
    ar=lang_code=="ar"
    if not len(df):return "لا توجد بلاغات ضمن الفلاتر الحالية." if ar else "No incidents match your current filters."
    ranks=sorted(data["district_counts"].items(),key=lambda x:x[1],reverse=True)
    toprisk=data["top_open_incidents"][:4]
    low=question.lower()
    if any(term in low for term in ["حي","أحياء","district","area","neigh","أكثر"]):
        lines=[f"- **{name}:** {count}" for name,count in ranks[:5]]
        return ("**الأحياء الأعلى في البلاغات الحالية:**\n" if ar else "**Districts with the most incidents:**\n")+"\n".join(lines)
    if any(term in low for term in ["خطر","حرج","risk","critical","priority","أولوية"]):
        lines=[f"- **{r['id']}** · {r['district']} · {r['severity']} · {r['priority']}/100" for r in toprisk]
        return (f"**البلاغات المفتوحة الحرجة:** {data['critical_open']}\n\n" if ar else f"**Open critical incidents:** {data['critical_open']}\n\n")+"\n".join(lines)
    if any(term in low for term in ["خطة","فرق","توزيع","recommend","plan","team","action","allocation","اقتراح"]):
        name=ranks[0][0]
        return (f"**مقترح تشغيلي استرشادي:**\n1. ابدأ بأعلى {len(toprisk)} بلاغات مفتوحة حسب الأولوية.\n2. خصص فريقًا إضافيًا لمراجعة البلاغات في **{name}**.\n3. راجع {data['critical_open']} بلاغًا حرجًا مفتوحًا وراقب زمن الاستجابة ({data['mean_response_hours']} ساعة في العينة).\n4. استخدم مختبر القرارات لتجربة سعات فرق مختلفة قبل أي قرار فعلي.\n\n*هذه توصيات حسابية على بيانات تجريبية، وليست أوامر تشغيل حقيقية.*" if ar else
            f"**Illustrative action plan:**\n1. Prioritize the top {len(toprisk)} open incidents by score.\n2. Review staffing in **{name}**, the most affected district.\n3. Examine {data['critical_open']} critical open incidents and response time ({data['mean_response_hours']} hours in this demo).\n4. Run alternative crew capacities in the Decision Lab.\n\n*This is a rule-based suggestion on synthetic data, not an operational order.*")
    return (f"**ملخص البيانات الحالية**\n- إجمالي البلاغات: **{data['filtered_count']}**\n- بلاغات مفتوحة: **{data['open_count']}**\n- بلاغات حرجة مفتوحة: **{data['critical_open']}**\n- مغلقة: **{data['resolved_count']}**\n- متوسط الاستجابة: **{data['mean_response_hours']} ساعة**\n- أكثر حي رصدًا: **{ranks[0][0]}** ({ranks[0][1]})\n\n*هذا تحليل قواعد مباشر لبيانات تجريبية. للحصول على إجابة لغوية مفتوحة اختَر Ollama المحلي.*" if ar else
            f"**Current data overview**\n- Total incidents: **{data['filtered_count']}**\n- Open: **{data['open_count']}**\n- Open critical: **{data['critical_open']}**\n- Resolved: **{data['resolved_count']}**\n- Average response: **{data['mean_response_hours']} hrs**\n- Most affected district: **{ranks[0][0]}** ({ranks[0][1]})\n\n*This is a rule-based analysis of synthetic data. Switch to local Ollama for open-ended generated answers.*")


def ask_ollama(prompt,model,df,lang_code,messages):
    if not model.strip():raise ValueError("Model is required")
    context=json.dumps(context_snapshot(df,lang_code),ensure_ascii=False)
    instructions=("You are Mirsad, a municipal operations analytic copilot. Use ONLY the provided synthetic data JSON for numerical facts. "
        "Never invent unprovided statistics. Clearly disclose that it is a synthetic demo and computed priority is rule-based. "
        "Answer the current user question clearly and concisely; suggestions are non-binding, do not claim real actions. "
        "Respond in Arabic." if lang_code=="ar" else
        "You are Mirsad, a municipal operations analytic copilot. Use ONLY the provided synthetic data JSON for numerical facts. "
        "Never invent unprovided statistics. Clearly disclose that this is a synthetic demo and computed priority is rule-based. "
        "Answer clearly and concisely, with non-binding suggestions only. Respond in English.")
    conversation=[{"role":"system","content":instructions+"\nCurrent filtered dataset summary:\n"+context}]
    for m in messages[-9:]:
        if m.get("role") in ("user","assistant"):
            conversation.append({"role":m["role"],"content":m["content"][:2500]})
    payload=json.dumps({"model":model.strip(),"messages":conversation,"stream":False,"options":{"temperature":0.25,"num_predict":600}}).encode("utf-8")
    request=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=payload,headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(request,timeout=90) as response:out=json.load(response)
    answer=out.get("message",{}).get("content","").strip()
    if not answer:raise ValueError("Empty model response")
    return answer


if chosen=="command":
    heading("OPERATIONS COMMAND",t("command_title"),t("command_sub"))
    opened=f[f.status!=2];n=len(f)
    c1,c2,c3,c4=st.columns(4)
    with c1:tile(t("records"),f"{n:,}",t("since"))
    with c2:tile(t("critical"),int((opened.severity==3).sum()),t("priority"),P["red"])
    with c3:tile(t("closed"),f"{(f.status==2).mean()*100:.0f}%" if n else "0%",t("since"),P["accent"])
    with c4:tile(t("response"),f"{f.response.mean():.1f} {t('hours')}" if n else "0",t("since"),P["blue"])
    left,right=st.columns([1.6,1],gap="large")
    with left:section(t("map"),t("map_sub"));st.plotly_chart(scatter_map(f),use_container_width=True,config={"scrollZoom":True,"displaylogo":False})
    with right:
        section(t("alerts"),t("alerts_sub"))
        top=priority(opened).sort_values("score",ascending=False).head(6)
        if top.empty:st.info(t("empty"))
        for _,r in top.iterrows():
            st.markdown(f'<div class="card" style="border-inline-start:4px solid {RISK_COLORS[int(r.severity)]};padding:12px 16px;margin-bottom:8px"><b>{html.escape(r.id)}</b> · <span style="color:{RISK_COLORS[int(r.severity)]}">{html.escape(sname(r.severity))}</span><div class="sub">{html.escape(dname(r.district))} · {html.escape(kname(r.kind))} · {t("score")}: {r.score:.1f}</div></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2,gap="large")
    with c1:
        section(t("risk_dist"));counts=f.severity.value_counts().reindex(range(4),fill_value=0)
        fig=go.Figure(go.Bar(x=counts.values,y=[sname(j) for j in counts.index],orientation="h",marker_color=[RISK_COLORS[j] for j in counts.index],text=counts.values,textposition="outside"))
        st.plotly_chart(style(fig,285),use_container_width=True)
    with c2:
        section(t("top_districts"));ct=f.district.value_counts().head(6).sort_values()
        fig=go.Figure(go.Bar(x=ct.values,y=[dname(j) for j in ct.index],orientation="h",marker_color=P["accent"],text=ct.values,textposition="outside"))
        st.plotly_chart(style(fig,285),use_container_width=True)
    section(t("latest"));st.dataframe(display_table(f.sort_values("detected",ascending=False).head(12),["id","district","kind","severity","status","detected","area"]),hide_index=True,use_container_width=True)

elif chosen=="explore":
    heading("INCIDENT INTELLIGENCE",t("explore_title"),t("explore_sub"))
    ranked=priority(f).sort_values("score",ascending=False)
    if ranked.empty:st.warning(t("empty"))
    else:
        ids=ranked.id.tolist();selected=st.selectbox(t("choose"),ids,format_func=lambda x:f"{x} · {dname(ranked.set_index('id').loc[x,'district'])}")
        r=ranked[ranked.id==selected].iloc[0]
        c1,c2,c3,c4=st.columns(4)
        with c1:tile(t("score"),f"{r.score}/100",t("priority"),P["gold"])
        with c2:tile(t("severity"),sname(r.severity),t("since"),RISK_COLORS[int(r.severity)])
        with c3:tile(t("area"),f"{r.area:,.1f}","m²")
        with c4:tile(t("confidence"),f"{r.confidence}%",t("demo"),P["blue"])
        a,b=st.columns([1.2,1],gap="large")
        with a:
            section(t("details"));st.markdown(f'<div class="card"><b>{r.id} · {html.escape(dname(r.district))}</b><div class="sub">{t("type")}: {html.escape(kname(r.kind))} · {t("status")}: {html.escape(stname(r.status))}<br>{t("team")}: {html.escape(teamname(r.team))}<br>{t("date")}: {r.detected:%Y-%m-%d %H:%M}<br>{t("lat")}: {r.lat:.5f} · {t("lon")}: {r.lon:.5f}</div></div>',unsafe_allow_html=True)
            st.plotly_chart(scatter_map(ranked[ranked.id==selected]),use_container_width=True)
        with b:
            section(t("factors"));fig=go.Figure(go.Bar(x=[r.population,r.facility,min(100,r.area/175*100)],y=[t("population"),t("facility"),t("size")],orientation="h",marker_color=P["accent"]))
            fig.update_xaxes(range=[0,110]);st.plotly_chart(style(fig,285),use_container_width=True)
            section(t("change"))
            with st.form("incident_update"):
                s=st.selectbox(t("status"),range(3),index=int(r.status),format_func=stname)
                team=st.selectbox(t("team"),range(5),index=int(r.team),format_func=teamname)
                if st.form_submit_button(t("save"),type="primary",use_container_width=True):
                    mask=st.session_state.records.id==selected
                    st.session_state.records.loc[mask,"status"]=s;st.session_state.records.loc[mask,"team"]=team
                    st.success(t("saved"));st.rerun()

elif chosen=="analytics":
    heading("ANALYTICS STUDIO",t("analytics_title"),t("analytics_sub"))
    if f.empty:st.warning(t("empty"))
    else:
        a,b=st.columns([1.4,1],gap="large")
        with a:
            section(t("trend"));ct=f.groupby(f.detected.dt.date).size().reset_index(name="count");ct.columns=["day","count"]
            fig=go.Figure(go.Scatter(x=ct.day,y=ct["count"],mode="lines+markers",line=dict(color=P["accent"],width=3),fill="tozeroy",fillcolor="rgba(37,184,146,.12)"))
            st.plotly_chart(style(fig,342),use_container_width=True)
        with b:
            section(t("cycle"));ct=f.status.value_counts().reindex(range(3),fill_value=0)
            fig=go.Figure(go.Pie(labels=[stname(j) for j in ct.index],values=ct.values,hole=.72,marker=dict(colors=[STATE_COLORS[j] for j in ct.index])))
            fig.update_layout(annotations=[dict(text=f"{len(f)}",showarrow=False,font=dict(size=26,color=P["text"]))]);st.plotly_chart(style(fig,342,True),use_container_width=True)
        a,b=st.columns([1.12,1],gap="large")
        with a:
            section(t("heat"));matrix=pd.crosstab(f.district.map(dname),f.kind.map(kname))
            fig=px.imshow(matrix,aspect="auto",text_auto=True,color_continuous_scale=[[0,P["panel"]],[.5,P["blue"]],[1,P["accent"]]])
            st.plotly_chart(style(fig,410),use_container_width=True)
        with b:
            section(t("resp_district"));s=f.groupby("district").response.mean().sort_values().tail(8)
            fig=go.Figure(go.Bar(x=s.values.round(1),y=[dname(j) for j in s.index],orientation="h",marker_color=P["blue"],text=s.values.round(1),textposition="outside"))
            st.plotly_chart(style(fig,410),use_container_width=True)
        section(t("export"));st.download_button(t("export_csv"),f.to_csv(index=False).encode("utf-8-sig"),"balmirsad_demo.csv","text/csv",type="primary")

elif chosen=="lab":
    heading("DECISION INTELLIGENCE",t("lab_title"),t("lab_sub"))
    a,b,c=st.columns(3)
    with a:ws=st.slider(t("size"),0,100,35,5)
    with b:wp=st.slider(t("population"),0,100,35,5)
    with c:wf=st.slider(t("facility"),0,100,30,5)
    if ws+wp+wf==0:st.warning("Select at least one weight / اختر وزنًا أكبر من صفر")
    ranked=priority(f[f.status!=2],ws,wp,wf).sort_values("score",ascending=False)
    a,b,c=st.columns(3)
    with a:tile(t("open"),len(ranked),t("since"))
    with b:tile(t("above80"),int((ranked.score>=80).sum()),t("priority"),P["red"])
    with c:tile(t("maxscore"),float(ranked.score.max()) if len(ranked) else 0,"/ 100",P["gold"])
    a,b=st.columns([1.3,1],gap="large")
    with a:section(t("ranking"));st.dataframe(display_table(ranked.head(20),["id","district","severity","score","area","status"]),hide_index=True,use_container_width=True,height=425)
    with b:
        section(t("dispatch"));teams=st.slider(t("teams"),1,12,5);capacity=st.slider(t("capacity"),1,8,3)
        st.metric(t("total_team"),teams*capacity);st.metric(t("backlog"),max(0,len(ranked)-teams*capacity))
        st.progress(min(1,(teams*capacity)/max(1,len(ranked))),text=t("coverage"))
        if st.button(t("dispatch_btn"),type="primary",use_container_width=True,disabled=ranked.empty):
            plan=ranked.head(teams*capacity)[["id","district","score"]].copy();plan["assigned_crew"]=[f"{t('team')} {j%teams+1:02d}" for j in range(len(plan))]
            st.session_state.dispatch=plan.to_dict("records");st.success(t("dispatch_done"))
        if st.session_state.dispatch:
            p=pd.DataFrame(st.session_state.dispatch)
            st.dataframe(p,hide_index=True,use_container_width=True,height=180)
            st.download_button(t("export_plan"),p.to_csv(index=False).encode("utf-8-sig"),"response_plan.csv","text/csv")
    section(t("formula"));st.info(t("formula_desc"))


# Floating chat: in every workspace, popover stays anchored bottom-left.
# Real LLM conversation requires a running local Ollama model; rules fallback is explicit.
if "ai_engine" not in st.session_state: st.session_state.ai_engine="local_ai"
if "ai_model_name" not in st.session_state: st.session_state.ai_model_name="qwen2.5:3b"

def submit_floating_prompt(query, current_df, current_lang):
    query=query.strip()
    if not query:return
    st.session_state.messages.append({"role":"user","content":query})
    mode=st.session_state.get("ai_engine","local_ai")
    if mode=="local_ai":
        try:
            answer=ask_ollama(query,st.session_state.get("ai_model_name","qwen2.5:3b"),current_df,current_lang,st.session_state.messages)
        except (OSError, ValueError, TimeoutError, KeyError) as exc:
            preface=("تعذّر الاتصال بنموذج Ollama المحلي؛ هذا رد تحليلي مباشر وليس رد ذكاء اصطناعي توليدي.\n\n" if current_lang=="ar" else "Local Ollama unavailable; showing rule-based insights, not a generative AI response.\n\n")
            answer=preface+instant_answer(query,current_df,current_lang)
    else:answer=instant_answer(query,current_df,current_lang)
    st.session_state.messages.append({"role":"assistant","content":answer})

with st.container(key="mirsad_floating_chat"):
    with st.popover("✦  اسأل المرصاد AI" if lang=="ar" else "✦  Ask Mirsad AI", use_container_width=False):
        st.markdown("### ✦ " + ("مساعد المرصاد الذكي" if lang=="ar" else "Mirsad AI Copilot"))
        st.caption(("محادثة مرتبطة ببيانات الداشبورد والفلاتر الحالية" if lang=="ar" else "Chat with your current dashboard data and filters"))
        engines=["local_ai","insights"]
        choice=st.segmented_control("AI Mode" if lang=="en" else "وضع المحادثة",engines,format_func=lambda x:("🧠 Ollama AI" if x=="local_ai" else ("تحليل مباشر" if lang=="ar" else "Instant insights")),key="ai_engine",default="local_ai")
        if choice=="local_ai":
            st.text_input("النموذج المحلي" if lang=="ar" else "Local model",key="ai_model_name")
            try:
                with urllib.request.urlopen("http://127.0.0.1:11434/api/tags",timeout=.6) as res:tags=json.load(res)
                installed=[i.get("name","") for i in tags.get("models",[])]
                if any(m==st.session_state.ai_model_name for m in installed):st.caption("🟢 " + ("النموذج جاهز للمحادثة" if lang=="ar" else "Model ready"))
                else:st.caption("🟠 " + ("شغّلي Ollama ونزّلي النموذج المحدد" if lang=="ar" else "Download the selected Ollama model"))
            except (OSError,ValueError):st.caption("🟠 " + ("شغّلي Ollama على جهازك لتفعيل المحادثة الذكية" if lang=="ar" else "Start Ollama locally for generative conversation"))
        else:st.caption("ⓘ " + ("إجابات قواعد وتحليل حسابي، بدون نموذج لغوي" if lang=="ar" else "Rule-based answers, not a language model"))
        with st.container(height=275,border=False):
            if not st.session_state.messages:
                st.info("أهلًا! اسأليني عن الأحياء، البلاغات الحرجة، أو اقترحي سيناريو توزيع الفرق." if lang=="ar" else "Hi! Ask me about districts, critical incidents, or response strategies.")
            for m in st.session_state.messages[-14:]:
                with st.chat_message(m["role"]):st.markdown(m["content"])
        with st.form("floating_ai_form",clear_on_submit=True,border=False):
            msg=st.text_input("رسالتك" if lang=="ar" else "Your message",placeholder=("اكتب سؤالك هنا..." if lang=="ar" else "Type your question..."),label_visibility="collapsed")
            submitted=st.form_submit_button("إرسال ↗" if lang=="ar" else "Send ↗",use_container_width=True,type="primary")
            if submitted and msg.strip():submit_floating_prompt(msg,f,lang);st.rerun()
        quick=st.columns(2)
        with quick[0]:
            if st.button("📊 " + ("ملخص" if lang=="ar" else "Summary"),key="ai_quick_summary",use_container_width=True):
                submit_floating_prompt(t("q1"),f,lang);st.rerun()
        with quick[1]:
            if st.button("⚠ " + ("الأولويات" if lang=="ar" else "Priorities"),key="ai_quick_risk",use_container_width=True):
                submit_floating_prompt(t("q3"),f,lang);st.rerun()
        if st.button("↻ "+t("clear_chat"),key="floating_clear",use_container_width=True):st.session_state.messages=[];st.rerun()
        st.caption("🔒 "+("البيانات تجريبية؛ وضع Ollama يعمل محليًا عند تشغيله." if lang=="ar" else "Synthetic data; Ollama runs locally when configured."))

st.markdown(f'<div class="footer">✦ BALMIRSAD · STREAMLIT COMMAND EXPERIENCE · {html.escape(t("source"))}</div>',unsafe_allow_html=True)
