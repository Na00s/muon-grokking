"""Final manuscript checks against the completed new controls and horizons."""
from pathlib import Path
import hashlib
import json
import math
import re
from fractions import Fraction
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC = ROOT/'work/muon-grokking/paper/source'
OLD = ROOT/'work/submission_audit/input'
PDF = HERE/'build/main.pdf'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

checks=[]
def check(name, passed, **details):
    checks.append(dict(name=name, passed=bool(passed), **details))
    assert passed, (name,details)

new_audits={}
for name,path in [
    ('fourier_primary',ROOT/'work/fourier_control/results/verification.json'),
    ('fourier_secondary',ROOT/'work/fourier_control/results/frequency_verification.json'),
    ('independent_primary',HERE/'control_audit/independent_verification.json'),
    ('independent_secondary',HERE/'control_audit/secondary_direct_verification.json'),
]:
    d=json.loads(path.read_text())
    check(name,d['all_passed'] and all(c['passed'] for c in d['checks']),checks=d['check_count'])
    new_audits[name]=dict(checks=d['check_count'],sha256=sha(path))

coverage=json.loads((HERE/'control_audit/coverage_bound.json').read_text())
p,m=coverage['p'],coverage['training_examples'];n=p*p
ratio=Fraction(math.comb(n-p,m),math.comb(n,m))
product=math.prod(Fraction(n-m-i,n-i) for i in range(p))
check('exact answer-orbit coverage identity',p==113 and m==3830 and ratio==product)
check('answer-orbit coverage numerical bound',math.isclose(float(ratio),coverage['single_orbit_uncovered_probability'],rel_tol=1e-13) and math.isclose(float(p*ratio),coverage['union_bound_any_uncovered'],rel_tol=1e-13) and float(p*ratio)<2.89e-16)
control_manifest=json.loads((HERE/'control_audit/audit_manifest.json').read_text())
for name,digest in control_manifest['source_hashes'].items():
    if name.startswith('work/fourier_control/') and (name.endswith('.py') or '/results/' in name):
        check('independently audited control input '+name,sha(ROOT/name)==digest)

long=ROOT/'work/long_horizon'
audit=json.loads((long/'verification.json').read_text())
check('long-horizon completed audit',audit['status']=='passed' and audit['passed']==audit['total'] and all(c['passed'] for c in audit['checks']),checks=audit['total'])
check('long-horizon audit matches runner and protocol',audit['runner_sha256']==sha(long/'run_long_horizon.py') and audit['protocol_sha256']==sha(long/'protocol.json'))
d=json.loads((long/'summary.json').read_text())
records=d['records']
check('six completed long horizons',d['status']=='completed' and len(records)==6 and all(r['end_step']==100000 for r in records))
check('six specified corrected branches',sorted((r['operation'],r['seed']) for r in records)==[('addition',seed) for seed in range(5)]+[('subtraction',0)])
check('zero-event interpretation matches records',all(r['first_joint_failure_step'] is None for r in records))
check('reported perfect addition endpoints',all(r['final_train_accuracy']==r['final_test_accuracy']==1 for r in records if r['operation']=='addition'))
check('420000 added training updates',sum(r['extension_updates'] for r in records)==420000)
check('five matched addition arms',[r['seed'] for r in records if r['operation']=='addition']==list(range(5)))
check('inherited arithmetic starts',all(r['arithmetic_start_step']==(6000 if r['operation']=='addition' else 0) for r in records))

figs=sorted(OLD.glob('figure*.pdf'))
check('six original figures preserved',len(figs)==6 and all(p.read_bytes()==(SRC/p.name).read_bytes() for p in figs))
table_files=[]
for p in sorted((OLD/'tables').glob('*.tex')):
    body=lambda t:re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}',t,re.S).group(0) if r'\begin{tabular}' in t else ''
    nums=lambda t:re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?',body(t))
    check('original table numerical cells '+p.name,nums(p.read_text())==nums((SRC/'tables'/p.name).read_text()))
    table_files.append(p.name)
check('17 original table files',len(table_files)==17)
check('original submission archive unchanged',sha(Path('/Users/alipro/Desktop/ICLR_Submission_Muon_Grokking.zip'))=='3d2c96208e191f9e73476f15606853e1d3cd43dc4b231b84fa2fd9cd2336b42d')

reader=PdfReader(PDF)
pages=[p.extract_text() for p in reader.pages]
text='\n'.join(pages)
reference_page=next(i+1 for i,t in enumerate(pages) if 'REFERENCES' in t)
aux=(HERE/'build/main.aux').read_text()
main_end_pages=re.findall(r'\\newlabel\{sec:main-end\}\{\{[^}]*\}\{(\d+)\}',aux)
check('scientific main-end marker recorded',len(main_end_pages)==1)
main_end_page=int(main_end_pages[0])
check('ICLR 2027 initial scientific main limit',1<=main_end_page<=9,scientific_main_last_page=main_end_page,limit=9)
check('references follow scientific main text',reference_page>main_end_page,references_start=reference_page)
main_source=(SRC/'main.tex').read_text()
check('main-end marker precedes exempt statements',main_source.index(r'\label{sec:main-end}')<main_source.index(r'\subsubsection*{Reproducibility statement}')<main_source.index(r'\subsubsection*{AI use statement}')<main_source.index(r'\bibliography'))
check('required AI use statement present',r'\subsubsection*{AI use statement}' in main_source and 'Generative AI tools assisted' in main_source and 'ai use statement' in re.sub(r'\s+',' ',text).lower())
official=json.loads((HERE/'control_audit/official_template_verification.json').read_text())
check('official ICLR 2027 template bytes',official['all_passed'] and len(official['checks'])==4 and all(item['byte_identical'] and sha(SRC/item['local'])==item['sha256'] for item in official['checks']))
check('original nonfinite observations retained',bool(re.search(r'23\s*,\s*514',text) and re.search(r'38\s*,\s*955',text)))
check('early filtered observation retained','95.14' in text)
check('new control figure rendered','Figure 9:' in text)
check('new orbit identity rendered','answer-conditioned' in text or 'answer-orbit' in text)
log=(HERE/'build/main.log').read_text()
check('no undefined references',not re.search(r'(undefined references|Citation.*undefined|Reference.*undefined)',log,re.I))
check('no overfull boxes','Overfull' not in log)
manuscript='\n'.join(p.read_text() for p in SRC.rglob('*.tex'))
check('no em dashes','\u2014' not in manuscript and '---' not in manuscript)
check('requested wording updates applied',r'\subsection{Muon updates remain nearly constant at small gradient norms}' in main_source and r'\subsection{Freezing embeddings and readout stabilizes the tested depth-1 runs}' in main_source and 'The role of numerical error and the parameter group responsible for collapse differ between the two architectures.' in main_source)
check('Section 6 filtered-accuracy wording applied',r'In \emph{projection masking}, the filtered representation achieves $100\%$ accuracy over the full operand grid. The full model reaches $45.85\%$.' in (SRC/'main.tex').read_text())
check('coverage bound stated in appendix',all(s in (SRC/'fourier_control_appendix.tex').read_text() for s in [r'\binom{N-p}{m}/\binom{N}{m}',r'pq<2.89\times10^{-16}']))
check('mean result promoted',r'\label{eq:followup-mean}' in (SRC/'main.tex').read_text() and r'\label{tab:mean-update}' in (SRC/'main.tex').read_text())
check('single-pair scope explicit',all(s in (SRC/'fourier_control_appendix.tex').read_text() for s in ['prime $p=113$','$280$','29.9945']))

provenance=json.loads((SRC/'figures/followup_figure_provenance.json').read_text())
check('updated horizon figure',provenance['main_panel_c']['end_step']==100000)
check('updated horizon figure label',r'"6,000 to 100,000\nupdates"' in (SRC/'figures/build_followup_figures.py').read_text())
check('figure accurate event count',provenance['main_panel_c']['accurate_failure_count']==sum(r['first_joint_failure_step'] is not None for r in records if r['operation']=='addition'))
for name,digest in provenance['outputs'].items():
    check('plotted asset '+name,sha(SRC/name)==digest)
for name,digest in provenance['source_sha256'].items():
    check('plotted input '+name,sha(ROOT/name)==digest)
check('Fourier control figure current',sha(SRC/'fourier_memorizer_control.pdf')==sha(ROOT/'work/fourier_control/fourier_memorizer_control.pdf'))

review=json.loads((HERE/'visual_review.json').read_text())
check('rendered PDF reviewed',review['pdf_sha256']==sha(PDF) and review['pages_reviewed']==len(pages) and review['all_passed'])
result=dict(all_passed=all(c['passed'] for c in checks), checks=checks, total_checks=len(checks),
    scope='New Fourier controls and long-horizon extensions verified; original results retained and their Fourier interpretation revised.',
    prior_new_experiment_audit=dict(checks=604,computed=598,manual=6,status='Previously passed; evidence and immutable input snapshot retained.'),
    new_control_audits=new_audits,long_horizon_audit=dict(checks=audit['total'],sha256=sha(long/'verification.json'),summary_sha256=sha(long/'summary.json'),runner_sha256=sha(long/'run_long_horizon.py'),protocol_sha256=sha(long/'protocol.json'),checkpoint_manifest_sha256=sha(long/'checkpoint_manifest.json')),
    additional_training_updates=420000,completed_horizon=100000,
    original_figure_pdfs_byte_identical=6,original_appendix_tables_numerical_cells_preserved=17,
    original_observed_nonfinite_terminations_retained=True,source_archive_unchanged=True,
    total_pdf_pages=len(pages),main_text_last_page=main_end_page,scientific_main_page_limit=9,references_start_page=reference_page,
    page_count_basis='Compiled sec:main-end label; reproducibility and AI use statements are excluded by the ICLR 2027 author guidelines.',
    author_guidelines='https://iclr.cc/Conferences/2027/AuthorGuidelines',required_ai_use_statement_present=True,
    official_template_verification=dict(source=official['source'],sha256=sha(HERE/'control_audit/official_template_verification.json'),all_passed=True),
    undefined_references=False,overfull_boxes=False,em_dashes=False,
    visual_review_complete=True,pdf_sha256=sha(PDF),
    source_sha256={p.relative_to(SRC).as_posix():sha(p) for p in sorted(SRC.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and 'plot_cache' not in p.parts},
    original_figure_sha256={p.name:sha(p) for p in figs},retained_result_table_files=table_files)
(HERE/'final_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(dict,list))},indent=2))
