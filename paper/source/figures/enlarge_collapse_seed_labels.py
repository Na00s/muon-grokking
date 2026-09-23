"""Enlarge labels in two original vector figures without changing plot geometry.

Requires pypdf and reportlab. Supply LiberationSerif-Regular.ttf and
LiberationSerif-Bold.ttf. Original figures are retained in legibility_originals.
All original non-text drawing operations remain intact. The empty bottom legend
strip is covered and its swatches are redrawn with more space between entries.
No plot panel is covered or rescaled. The PDF canvas dimensions are unchanged.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, io, json, math, re
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, TextStringObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

BLACK=(.102,.102,.102)
BLUE=(.1216,.3725,.6588)
RED=(.7098,.1961,.2275)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def text_runs(page):
    state={};out=[]
    fonts=page['/Resources']['/Font']
    def visit(op,args,cm,tm):
        if op==b'rg':state['color']=tuple(float(v) for v in args)
        if op==b'Tf':state.update(font=str(args[0]),size=float(args[1]))
        if op==b'Tj':
            raw=args[0].get_original_bytes() if hasattr(args[0],'get_original_bytes') else bytes(args[0])
            values=[int.from_bytes(raw[i:i+2],'big') for i in range(0,len(raw),2)]
            text=''.join(chr(183 if v==121 else v+29) for v in values)
            a,b,c,d,e,f=cm
            out.append(dict(text=text,x=tm[4]*a+tm[5]*c+e,y=tm[4]*b+tm[5]*d+f,
                            size=state['size']*math.hypot(a,b),color=state['color'],
                            bold='Bold' in fonts[state['font']]['/BaseFont'],rotated=abs(b)>.1))
    page.extract_text(visitor_operand_before=visit)
    return out

class Overlay:
    def __init__(self,w,h,size):
        self.buffer=io.BytesIO();self.canvas=canvas.Canvas(self.buffer,pagesize=(w,h),invariant=1)
        self.size=size;self.labels=[]
    def text(self,text,x,y,anchor='left',bold=False,color=BLACK,rotate=0):
        c=self.canvas;font='FigureSerifBold' if bold else 'FigureSerif'
        self.labels.append(dict(text=text,x=x,y=y,size=self.size,anchor=anchor,bold=bold,color=color,rotation=rotate))
        c.saveState();c.setFillColorRGB(*color);c.setFont(font,self.size);c.translate(x,y);c.rotate(rotate)
        if anchor=='right':c.drawRightString(0,0,text)
        elif anchor=='center':c.drawCentredString(0,0,text)
        else:c.drawString(0,0,text)
        c.restoreState()
    def legend(self,w,entries,y=16):
        c=self.canvas;c.setFillColorRGB(1,1,1);c.rect(0,0,w,28,fill=1,stroke=0)
        widths=[pdfmetrics.stringWidth(t,'FigureSerif',self.size) for t,_,_ in entries]
        total=sum(widths)+30*len(entries)+12*(len(entries)-1);x=(w-total)/2
        assert x>=0
        for width,(text,color,dash) in zip(widths,entries):
            c.saveState();c.setStrokeColorRGB(*color);c.setLineWidth(1.8);c.setDash(dash)
            c.line(x,y+4,x+23,y+4);c.restoreState()
            self.text(text,x+30,y);x+=30+width+12
    def finish(self):self.canvas.save();return PdfReader(self.buffer).pages[0]

def collapse(runs,w,h):
    o=Overlay(w,h,14)
    for r in runs:
        t,x,y=r['text'],r['x'],r['y']
        if re.fullmatch(r'[01](?:\.\d+)?',t) and x<400 and r['size']>9:
            o.text(t,51.98 if x<100 else 387.65,y-1.3,anchor='right')
        elif re.fullmatch(r'\d\d,\d\d\d',t):
            oldwidth=pdfmetrics.stringWidth(t,'FigureSerif',r['size']);center=x+oldwidth/2
            if t=='45,000':o.text(t,w-2,y-1,anchor='right')
            else:o.text(t,center,y-1,anchor='center')
        elif t=='training step':
            center=x+pdfmetrics.stringWidth(t,'FigureSerif',r['size'])/2
            o.text(t,center,35,anchor='center')
        elif t=='spectral similarity':o.text(t,18,237.5,anchor='center',rotate=90)
        elif t=='accuracy':o.text(t,18,128,anchor='center',rotate=90)
        elif t in ('Collapse','Recovery'):
            center=x+pdfmetrics.stringWidth(t,'FigureSerif',r['size'])/2
            o.text(t,center,y,anchor='center')
        elif t=='0.55':o.text(t,x,y,bold=True,color=r['color'])
    # Preserve the leader-side right edge of the two paired-value annotations.
    for first,last,right,y,color in [('1.0000','0.9899',303.04,246.87,BLUE),('19.04%','24.78%',312.04,110.37,(.7529,.2235,.1686))]:
        spans=[(first,True),(' / ',False),(last,True)]
        widths=[pdfmetrics.stringWidth(t,'FigureSerifBold' if b else 'FigureSerif',14) for t,b in spans]
        x=right-sum(widths)
        for (t,b),width in zip(spans,widths):o.text(t,x,y,bold=b,color=color);x+=width
    o.legend(w,[('Top-frequency Jaccard',BLUE,[]),('Addition-family power cosine',(.5412,.7216,.8706),[7,5]),('Addition-only accuracy',(.8784,.5098,.0784),[2,2]),('Test accuracy',(.7529,.2235,.1686),[])])
    return o

def seeds(runs,w,h):
    o=Overlay(w,h,15)
    for r in runs:
        t,x,y=r['text'],r['x'],r['y']
        if t in ('0','0.5','1') and x<42:o.text(t,40.95,y-1.7,anchor='right')
        elif (t=='0' and x>42) or re.fullmatch(r'\d\d,\d\d\d',t):
            center=x+pdfmetrics.stringWidth(t,'FigureSerif',r['size'])/2
            o.text(t,center,y-1.7,anchor='center')
        elif t=='training step':
            center=x+pdfmetrics.stringWidth(t,'FigureSerif',r['size'])/2
            o.text(t,center,44,anchor='center')
        elif t=='test accuracy':o.text(t,13,277,anchor='center',rotate=90)
        elif t.startswith('seed '):o.text(t,85,y,bold=True,color=r['color'])
        elif 'evaluations below' in t:o.text(t,135.75,y,color=r['color'])
        elif '·' in t:o.text(t,377,y,color=r['color'])
        elif t.startswith('0 below'):o.text(t,404,y,color=r['color'])
        elif t=='16.3%':o.text(t,x,y,color=r['color'])
    o.legend(w,[('shared, before the freeze',(.4784,.4784,.4784),[]),('Muon continues',RED,[]),('embeddings and readout frozen',BLUE,[])])
    return o

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font',type=Path,required=True);parser.add_argument('--bold-font',type=Path,required=True)
    args=parser.parse_args();here=Path(__file__).resolve().parent
    pdfmetrics.registerFont(TTFont('FigureSerif',str(args.font)));pdfmetrics.registerFont(TTFont('FigureSerifBold',str(args.bold_font)))
    records=[]
    for name,builder,scale in [('figure1_collapse.pdf',collapse,.95*396/750),('figure3_seeds.pdf',seeds,.84*396/705.12)]:
        original=here/'legibility_originals'/name;out=here.parent/name
        reader=PdfReader(original);writer=PdfWriter();writer.add_page(reader.pages[0]);page=writer.pages[0]
        runs=text_runs(page);w,h=float(page.mediabox.width),float(page.mediabox.height)
        stream=ContentStream(page.get_contents(),writer);changed=[]
        for i,(args_,op) in enumerate(stream.operations):
            if op==b'Tj':stream.operations[i]=([TextStringObject('')],op);changed.append(i)
        assert len(changed)==len(runs)
        # Compare every untouched operator before adding the separate label overlay.
        previous=ContentStream(reader.pages[0].get_contents(),reader)
        assert all(a==b for i,(a,b) in enumerate(zip(previous.operations,stream.operations)) if i not in changed)
        page.replace_contents(stream);overlay=builder(runs,w,h)
        oldchars=Counter(''.join(''.join(r['text'].split()) for r in runs))
        newchars=Counter(''.join(''.join(r['text'].split()) for r in overlay.labels))
        assert oldchars==newchars,(name,oldchars-newchars,newchars-oldchars)
        oldnumbers=Counter(re.findall(r'\d+(?:[,.]\d+)*(?:%)?',' '.join(r['text'] for r in runs)))
        newnumbers=Counter(re.findall(r'\d+(?:[,.]\d+)*(?:%)?',' '.join(r['text'] for r in overlay.labels)))
        assert oldnumbers==newnumbers
        page.merge_page(overlay.finish());writer.add_metadata({'/Title':name.removesuffix('.pdf').replace('_',' '),'/Subject':'Enlarged labels; original plotted data and vector geometry preserved'})
        with out.open('wb') as f:writer.write(f)
        records.append(dict(figure=name,original='legibility_originals/'+name,original_sha256=sha(original),output_sha256=sha(out),canvas_pt=[w,h],canvas_unchanged=True,minimum_label_size_pt=overlay.size,minimum_printed_label_size_pt=overlay.size*scale,include_width_factor=scale,all_original_nontext_operations_unchanged=True,covered_area='Bottom legend strip only, y=0..28 PDF points',legend_swatches_redrawn=True,all_label_characters_and_numeric_tokens_preserved=True,text_show_operations_replaced=len(changed),labels=overlay.labels))
    result=dict(figures=records,font_sha256=sha(args.font),bold_font_sha256=sha(args.bold_font))
    (here/'collapse_seed_legibility_provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='labels'} for r in records],indent=2))
if __name__=='__main__':main()
