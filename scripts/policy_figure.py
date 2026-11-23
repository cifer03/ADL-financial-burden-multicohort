"""Build the candidate policy figure from aggregate output in the audit package."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

base=Path('review')
source=Path('aggregate/review/three_cohort_slots.csv')
data=pd.read_csv(base/source.name)
cohorts=['HRS','SHARE','KLoSA']
rules=['spending','spending divided by income-group median']
metrics=['lower-income cases','higher-income cases','all cases']
colors=['#205472','#ab532d','#dfe4e6']
def number(x):
    return f'{x:.1f}'.replace('.','·').replace('-','−')
plt.rcParams.update({'pdf.fonttype':42,'font.size':9,'font.family':'DejaVu Sans'})
fig=plt.figure(figsize=(10.2,5.8))
fig.text(.035,.955,'Who receives the 150 financial assessments?',fontsize=17,weight='bold')
fig.text(.035,.91,'Retrospective allocation per 1000 eligible observations with newly reported activity limitations',fontsize=10)
left=fig.add_axes([.21,.255,.32,.49]); right=fig.add_axes([.66,.255,.30,.49])
fig.text(.035,.82,'A   Allocation of assessments',fontsize=12,weight='bold')
fig.text(.578,.82,'B   Change in high-burden cases identified',fontsize=12,weight='bold')
for i,name in enumerate(cohorts):
    z=data[data.cohort.eq(name)]
    y=6-3*i
    for j,rule in enumerate(rules):
        p=z[z.rule.eq(rule)].set_index('assessed_per_1000_by_group')
        amounts=[p.loc[m,'estimate'] for m in metrics[:2]]+[150-p.loc['all cases','estimate']]
        np.testing.assert_allclose(sum(amounts),150)
        start=0
        for k,value in enumerate(amounts):
            left.barh(y-j,value,left=start,height=.78,color=colors[k])
            left.text(start+value/2,y-j,number(value),ha='center',va='center',color='white' if k<2 else '#29343b',fontsize=8)
            start+=value
    left.text(-.51,y-.45,name,va='center',weight='bold',transform=left.get_yaxis_transform())
    for k,(m,shade,marker) in enumerate(zip(metrics,[*colors[:2],'#292f36'],['o','s','D'])):
        row=p.loc[m];v=row.change_vs_spending
        right.errorbar(v,y+.25-.72*k,xerr=[[v-row.lower95_change],[row.upper95_change-v]],fmt=marker,color=shade,capsize=3,ms=5)
    right.text(-.22,y-.45,name,va='center',weight='bold',transform=right.get_yaxis_transform())
left.set(yticks=[6,5,3,2,0,-1],yticklabels=['Spending','With income']*3,xticks=[0,50,100,150],xlim=(0,150),xlabel='Assessments per 1000')
right.set(yticks=[],xticks=[-30,-15,0,15,30,45],xlim=(-34,47),xlabel='Change per 1000 (95% CI)')
right.axvline(0,color='#858a90',lw=.8,zorder=0)
for ax in [left,right]:
    ax.set_ylim(-1.8,6.85)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
fig.legend(handles=[Patch(color=colors[0],label='Lower-income cases'),Patch(color=colors[1],label='Higher-income cases'),Patch(color=colors[2],label='Below or at threshold (A)'),plt.Line2D([],[],color='#292f36',marker='D',ls='',label='All cases (B)')],loc='lower left',bbox_to_anchor=(.03,.11),ncol=4,frameon=False,fontsize=9)
fig.text(.035,.074,'High burden: household out-of-pocket medical spending above 10% of annual household income.',fontsize=9)
fig.text(.035,.038,'With income: spending divided by baseline median household income for the person’s income group.',fontsize=9)
fig.savefig(base/'figure_2.pdf')
