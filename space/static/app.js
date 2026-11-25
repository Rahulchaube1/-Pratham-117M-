import { Tokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.1.3/+esm';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs';
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const base=document.body.dataset.base;
const log=document.querySelector('#log'), status=document.querySelector('#status'),send=document.querySelector('button'),box=document.querySelector('textarea');
let session,tokenizer,history='';
function line(who,msg){const row=document.createElement('p');const label=document.createElement('strong');label.textContent=who+' ';row.append(label,document.createTextNode(msg));log.append(row);return row}
async function init(){status.textContent='Loading tokenizer and model (~148 MB on first use)…';
const r=await fetch(base+'tokenizer.json');if(!r.ok)throw Error('Tokenizer download failed');
tokenizer=new Tokenizer(await r.json(),{});session=await ort.InferenceSession.create(base+'chat.int8.onnx',{executionProviders:['wasm']});status.textContent='Ready';send.disabled=false}
document.querySelector('form').onsubmit=async(e)=>{e.preventDefault();const q=box.value.trim();if(!q||send.disabled)return;send.disabled=true;box.value='';line('You',q);const row=line('Pratham','…');let out=[];
try{const prompt=history+'User: '+q+'\nAssistant:';let ids=Array.from(tokenizer.encode(prompt).ids,Number).slice(-128);for(let i=0;i<32;i++){const {scores}=await session.run({ids:new ort.Tensor('int64',BigInt64Array.from(ids,BigInt),[1,ids.length])});const z=scores.data;const top=Array.from(z.keys()).sort((a,b)=>z[b]-z[a]).slice(0,20);let weights=top.map(j=>Math.exp((z[j]-z[top[0]])/.8));let t=Math.random()*weights.reduce((a,b)=>a+b,0),j=0;while(j<top.length-1&&(t-=weights[j])>0)j++;const next=top[j];if(next===0)break;out.push(next);ids.push(next);row.textContent='Pratham  '+tokenizer.decode(out);await new Promise(resolve=>setTimeout(resolve,0))}const reply=tokenizer.decode(out).split('\nUser:')[0].trim();history=prompt+' '+reply+'\n';row.textContent='Pratham  '+(reply||'[End of reply]');status.textContent='Ready'}catch(err){row.textContent='Generation failed: '+err.message}finally{send.disabled=false}};
init().catch(err=>status.textContent='Loading failed: '+err.message);
