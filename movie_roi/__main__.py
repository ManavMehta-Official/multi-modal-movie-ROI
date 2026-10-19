import argparse
import json
from .model import train,predict

def main():
    parser=argparse.ArgumentParser(description='Movie ROI research pipeline. Only load trusted locally trained artifacts.')
    sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--data',default='data/movies_multimodal.csv');t.add_argument('--output',default='artifacts');t.add_argument('--min-poster-coverage',type=float,default=.8,help='Abort if fewer than this fraction of rows have poster files (default 0.8). Use 0 only for an explicitly missing-image experiment.')
    t.add_argument('--regression-specs',help='JSON mapping each modality to a preselected regression specification.');t.add_argument('--reused-benchmark',action='store_true',help='Label evaluation as reused exploratory benchmark, not independent testing.')
    p=sub.add_parser('predict');p.add_argument('--input',required=True);p.add_argument('--artifacts',default='artifacts');p.add_argument('--poster');p.add_argument('--model',choices=['multimodal','tabular','tabular_text','tabular_poster'])
    args=parser.parse_args()
    result=train(args.data,args.output,min_poster_coverage=args.min_poster_coverage,regression_specs=json.load(open(args.regression_specs)) if args.regression_specs else None,reused_benchmark=args.reused_benchmark) if args.command=='train' else predict(json.load(open(args.input)),args.artifacts,args.poster,args.model)
    print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
