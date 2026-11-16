import importlib.util,io,base64,zlib,json,hashlib
from pathlib import Path
p=Path('/Users/jnolascodecarles/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/5F74DB0A-DEAF-4F1F-9792-F7BCFD3616D7/predictor_v1_Grupo_9.py')
s=importlib.util.spec_from_file_location('entrega',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class CatState:
 def __setstate__(self,state):self.state=state
class Inspect(m._ModelUnpickler):
 def find_class(self,module,name):
  if module.startswith('catboost'):return CatState
  return super().find_class(module,name)
packed=base64.b64decode(m._MODEL_DATA);assert hashlib.sha256(packed).hexdigest()==m._PAYLOAD_SHA256
b=Inspect(io.BytesIO(zlib.decompress(packed))).load()
out=dict(metadata=b.metadata,weights=b.weights,threshold=b.threshold,pipelines=[])
def inspect(obj):
 r={'class':type(obj).__name__}
 if hasattr(obj,'get_params'):r['params']={k:v for k,v in obj.get_params(deep=False).items() if isinstance(v,(str,int,float,bool,type(None)))}
 for attr in ['spec_','feature_names_in_','variant','branch','cats_']:
  if hasattr(obj,attr):r[attr]=getattr(obj,attr)
 for attr in ['base_','schema_','model_']:
  if hasattr(obj,attr):r[attr]=inspect(getattr(obj,attr))
 if isinstance(obj,CatState):
  r['state_keys']=list(obj.state)
  r['init_params']=obj.state.get('_init_params');r['test_evals']=obj.state.get('_test_evals')
 return r
for pipeline in b.pipelines:out['pipelines'].append({name:inspect(obj) for name,obj in pipeline.steps})
def convert(x):
 if hasattr(x,'tolist'):return x.tolist()
 return str(x)
text=json.dumps(out,default=convert,indent=2);Path('experiments/comparacion_entrega_v1/estructura.json').write_text(text);print(text)
