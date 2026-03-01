#!/usr/bin/env python3
import argparse, json, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from joblib import load

def main():
  ap=argparse.ArgumentParser();
  ap.add_argument('--data', required=True);
  ap.add_argument('--label', required=True);
  ap.add_argument('--model', required=True)
  ap.add_argument('--threshold', type=float, default=0.5);
  ap.add_argument('--out-json', default='reports/dt_test_metrics.json');
  ap.add_argument('--out-md', default='reports/dt_test_summary.md');
  ap.add_argument('--out-cm-png', default='reports/dt_confusion_matrix.png')
  args=ap.parse_args()

  df = pd.read_csv(args.data)
  clf = load(args.model)
  os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

  y=df[args.label].values
  X=df.drop(columns=[args.label])

  proba=clf.predict_proba(X)[:,1] if hasattr(clf,'predict_proba') else (lambda d:(d-d.min())/(d.max()-d.min()+1e-9))(clf.decision_function(X))

  ypred= (proba >= args.threshold).astype(int)

  auc=roc_auc_score(y, proba)
  acc=accuracy_score(y, ypred)
  cm=confusion_matrix(y, ypred)

  json.dump({'auc':float(auc),'accuracy':float(acc),'threshold':args.threshold,'confusion_matrix':cm.tolist()}, open(args.out_json,'w'), indent=2)

  md=f"""# Decision Tree — Test Metrics\n\n- AUC: {auc:.4f}\n- Accuracy: {acc:.4f}\n- Threshold: {args.threshold:.2f}\n\n**Confusion Matrix** (rows=Actual, cols=Predicted):\n{cm.tolist()}\n"""; open(args.out_md,'w').write(md)
  fig,ax=plt.subplots(figsize=(4,4))
  im=ax.imshow(cm, cmap='Greens')
  [ax.text(j,i,str(v),ha='center',va='center') for (i,j),v in np.ndenumerate(cm)]; ax.set_title('Confusion Matrix — Decision Tree'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); fig.colorbar(im, ax=ax, fraction=0.046,pad=0.04); plt.tight_layout(); plt.savefig(args.out_cm_png, dpi=150)
  print(md)

if __name__=='__main__': main()
