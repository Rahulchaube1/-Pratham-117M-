import {AutoModelForCausalLM,AutoTokenizer,LogitsProcessor,LogitsProcessorList,TextStreamer,env} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.1/dist/transformers.min.js';
env.allowLocalModels=false;
const status=document.querySelector('#status'),form=document.querySelector('form'),send=form.querySelector('button'),box=form.querySelector('textarea'),log=document.querySelector('#log');
const cfg=await (await fetch('./release.json')).json();
let mdl,tok,processors,pending;const history=[];
class PrathamCorrection extends LogitsProcessor {
 constructor(weights){super();this.weights=weights}
 _call(ids,scores){const z=scores.data;const x=cfg.anchors.map(i=>z[i]);if(this.weights.length!==z.length*x.length)throw Error('Adapter vocabulary mismatch');
 for(let i=0;i<z.length;i++){let delta=0;for(let j=0;j<x.length;j++)delta+=this.weights[i*x.length+j]*x[j];z[i]+=delta}return scores}
}
async function load(){
 const progress=e=>{if(e.status==='progress'&&e.total)status.textContent='Loading '+e.file+' · '+Math.round(e.loaded/e.total*100)+'%'};
 const id=cfg.onnx,opts={revision:cfg.onnx_revision,progress_callback:progress};
 tok=await AutoTokenizer.from_pretrained(id,opts);
 const useGPU=!!await navigator.gpu?.requestAdapter();
 try{mdl=await AutoModelForCausalLM.from_pretrained(id,{...opts,device:useGPU?'webgpu':'wasm',dtype:'q4'})}
 catch(err){if(!useGPU)throw err;mdl=await AutoModelForCausalLM.from_pretrained(id,{...opts,device:'wasm',dtype:'q4'})}
 const raw=await fetch('./adapter.f32');if(!raw.ok)throw Error('Adapter download failed');
 processors=new LogitsProcessorList();processors.push(new PrathamCorrection(new Float32Array(await raw.arrayBuffer())))}
function row(who,message){const p=document.createElement('p');const tag=document.createElement('strong');tag.textContent=who+' ';const t=document.createElement('span');t.textContent=message;p.append(tag,t);log.append(p);return t}
form.onsubmit=async e=>{e.preventDefault();let q=box.value.trim();if(!q||send.disabled)return;send.disabled=true;row('You',q);box.value='';const answer=row('Pratham','');
try{if(!mdl||!processors){status.textContent='Loading the model on this device…';await load()}
 const messages=[...history.slice(-8),{role:'user',content:q}];const input=tok.apply_chat_template(messages,{add_generation_prompt:true,return_dict:true});
 if(input.input_ids.dims[1]>2048)throw Error('This conversation exceeds 2,048 input tokens; shorten the message or reload for a new chat.');
 status.textContent='Generating…';let text='';const streamer=new TextStreamer(tok,{skip_prompt:true,skip_special_tokens:true,callback_function:s=>{text+=s;answer.textContent=text}});
 const ids=await mdl.generate({...input,logits_processor:processors,streamer,do_sample:false,max_new_tokens:128});
 const reply=tok.decode(ids[0].tolist().slice(input.input_ids.dims[1]),{skip_special_tokens:true}).trim();answer.textContent=reply||'[End of reply]';history.push({role:'user',content:q},{role:'assistant',content:reply});status.textContent='Ready'
 }catch(err){answer.textContent='';status.textContent='Could not generate: '+err.message}finally{send.disabled=false}};
status.textContent='First message downloads the model. A desktop browser with WebGPU is recommended.';send.disabled=false;
