"""Recompute Figure 3 and check displayed values against the manuscript."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True, help='directory containing the CHARLS and KHP ZIPs')
    p.add_argument('--elsa',type=Path,required=True,help='directory containing extracted UKDS 5050 Stata files')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent
    expected=pd.read_csv(here/'reference/figure3_checks.csv')
    actual=[]
    for name,source in [('charls',args.inputs),('elsa',args.elsa),('khp',args.inputs)]:
        target=args.output/(name+'.csv')
        subprocess.run([sys.executable,str(here/'complementary'/(name+'.py')),str(source),str(target)],check=True)
        df=pd.read_csv(target)
        if 'model' not in df: df['model']='primary'
        df['source']=name
        actual.append(df)
    result=pd.concat(actual,ignore_index=True)
    keys=['source','outcome','model']
    check=expected.merge(result,on=keys,validate='one_to_one',suffixes=('_ref',''),how='outer',indicator=True)
    assert check._merge.eq('both').all()
    for r in check.itertuples():
        assert r.n==r.n_ref
        for field in ['estimate','lower95','upper95']:
            assert round(getattr(r,field),r.decimals)==getattr(r,field+'_ref'),(r.outcome,field)
    display=check[check.model!='followup_weighted'].reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(9,4.1))
    fig.subplots_adjust(left=.43,right=.76,bottom=.16,top=.87)
    for i,r in display.iterrows():
        y=len(display)-i
        ax.errorbar(r.estimate,y,xerr=[[r.estimate-r.lower95],[r.upper95-r.estimate]],fmt='o',color='#205472',capsize=3)
        ax.text(1.02,y,f'{r.estimate:.{r.decimals}f} ({r.lower95:.{r.decimals}f} to {r.upper95:.{r.decimals}f})',transform=ax.get_yaxis_transform(),va='center')
    ax.set_yticks(np.arange(len(display),0,-1),display.label)
    ax.set_ylim(.4,len(display)+.6);ax.axvline(0,color='grey',lw=.8)
    ax.set_xlabel('Difference (percentage points, 95% CI)')
    ax.spines[['right','top','left']].set_visible(False)
    fig.suptitle('Household budget burden and unmet care after functional limitations')
    fig.savefig(args.output/'figure_3.pdf');plt.close(fig)
    print('All six plotted contrasts and both ELSA sensitivity estimates passed the manuscript checks.')
if __name__=='__main__':
    main()
