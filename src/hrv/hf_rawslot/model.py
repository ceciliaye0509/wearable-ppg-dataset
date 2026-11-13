"""Launch the unchanged colleague model against the HF raw-slot dataset."""
from pathlib import Path
import subprocess,sys,os,argparse

def main():
    root=Path(__file__).resolve().parents[2]/'model_baselines'
    script=root/'supervised/main_supervised_group4_kazemi_decoder_score.py'
    arguments=sys.argv[1:].copy()
    if '--segment-dataset-manifest' in arguments:
        raise SystemExit('Use hrv.v46.model for segmented data; this entry is HF raw-slot only.')
    if '--help' not in arguments and '-h' not in arguments:
        parser=argparse.ArgumentParser(add_help=False)
        parser.add_argument('--data_dir',required=True)
        parser.add_argument('--task',required=True)
        parser.parse_known_args(arguments)
    # Keep caller-relative dataset paths valid when the colleague runner uses
    # its original working directory. Its source and parser remain untouched.
    for i,arg in enumerate(arguments):
        if arg=='--data_dir' and i+1<len(arguments):arguments[i+1]=str(Path(arguments[i+1]).expanduser().resolve())
        elif arg.startswith('--data_dir='):arguments[i]='--data_dir='+str(Path(arg.split('=',1)[1]).expanduser().resolve())
    env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH',''))
    raise SystemExit(subprocess.call([sys.executable,str(script),*arguments],cwd=root,env=env))

if __name__=='__main__':main()
