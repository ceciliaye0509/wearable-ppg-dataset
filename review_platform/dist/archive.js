/* Read a reviewer ZIP locally. No network calls or third-party libraries. */
(function(root){
  'use strict';
  const limit=300*1024*1024,decode=new TextDecoder('utf-8',{fatal:true});
  const fail=message=>{throw Error(message);};
  const table=Array.from({length:256},(_,n)=>{for(let j=0;j<8;j++)n=n&1?0xedb88320^(n>>>1):n>>>1;return n>>>0;});
  function crc32(bytes){let c=0xffffffff;for(const b of bytes)c=table[(c^b)&255]^(c>>>8);return(c^0xffffffff)>>>0;}
  function safe(name){return name&&!name.startsWith('/')&&!name.includes('\\')&&!name.includes('\0')&&!name.split('/').some(p=>p==='..'||p==='.')&&!/^[A-Za-z]:/.test(name);}
  async function inflate(bytes,length){
    let stream;try{stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));}catch{fail('ZIP decompression is unavailable. Unzip the package and open its folder instead.');}
    const reader=stream.getReader(),parts=[];let size=0;
    try{while(true){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>length)fail('ZIP entry expands beyond its declared size.');parts.push(value);}}catch(e){await reader.cancel().catch(()=>{});throw e;}
    if(size!==length)fail('ZIP entry length is incorrect.');const out=new Uint8Array(size);let i=0;for(const part of parts){out.set(part,i);i+=part.length;}return out;
  }
  async function files(file){
    if(!file.size||file.size>limit)fail('Reviewer ZIP must be at most 300 MB.');
    const bytes=new Uint8Array(await file.arrayBuffer()),v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);let end=-1;
    for(let i=bytes.length-22;i>=Math.max(0,bytes.length-65557);i--)if(v.getUint32(i,true)===0x06054b50&&i+22+v.getUint16(i+20,true)===bytes.length){end=i;break;}
    if(end<0)fail('ZIP directory not found.');
    const count=v.getUint16(end+10,true),start=v.getUint32(end+16,true),size=v.getUint32(end+12,true);
    if(v.getUint16(end+4,true)||v.getUint16(end+6,true)||v.getUint16(end+8,true)!==count||count>10000||count===65535||start+size!==end)fail('Unsupported or damaged ZIP directory.');
    const entries=[],seen=new Set(),ranges=[];let p=start,total=0;
    for(let j=0;j<count;j++){
      if(p+46>end||v.getUint32(p,true)!==0x02014b50)fail('Damaged ZIP entry.');
      const flags=v.getUint16(p+8,true),method=v.getUint16(p+10,true),crc=v.getUint32(p+16,true),packed=v.getUint32(p+20,true),length=v.getUint32(p+24,true),nl=v.getUint16(p+28,true),el=v.getUint16(p+30,true),cl=v.getUint16(p+32,true),off=v.getUint32(p+42,true);
      if(p+46+nl+el+cl>end||v.getUint16(p+34,true)||(flags&~0x808)||![0,8].includes(method)||length>25*1024*1024||((v.getUint32(p+38,true)>>>16)&0xf000)===0xa000)fail('Encrypted, linked, oversized or unsupported ZIP entry.');
      const name=decode.decode(bytes.subarray(p+46,p+46+nl));if(!safe(name)||seen.has(name))fail('Unsafe or duplicate ZIP path.');seen.add(name);p+=46+nl+el+cl;
      if(off+30>start||v.getUint32(off,true)!==0x04034b50)fail('Invalid ZIP file offset.');
      const localName=v.getUint16(off+26,true),extra=v.getUint16(off+28,true),begin=off+30+localName+extra,stop=begin+packed;
      if(begin>start||stop>start||decode.decode(bytes.subarray(off+30,off+30+localName))!==name||v.getUint16(off+6,true)!==flags||v.getUint16(off+8,true)!==method||ranges.some(([a,b])=>off<b&&stop>a))fail('Inconsistent or overlapping ZIP entries.');
      ranges.push([off,stop]);total+=length;if(total>limit)fail('Expanded ZIP exceeds 300 MB.');
      if(!name.endsWith('/')&&!name.startsWith('__MACOSX/')&&!name.split('/').some(x=>x.startsWith('.')))entries.push({name,method,crc,length,begin,stop});
    }
    if(p!==end)fail('Unexpected ZIP directory content.');
    const out=[];
    for(const e of entries){
      if(!/\.(ecgreview|json|json\.gz)$/.test(e.name))continue;
      const packed=bytes.subarray(e.begin,e.stop),data=e.method===0?packed:await inflate(packed,e.length);
      if(data.length!==e.length||crc32(data)!==e.crc)fail('ZIP checksum failed: '+e.name);
      const f=new File([data],e.name.split('/').at(-1));f._relative=e.name;out.push(f);
    }
    if(!out.some(f=>f.name.endsWith('.ecgreview')||f.name==='history_manifest.json'))fail('This ZIP has no .ecgreview descriptor. Annotation-only archives need the research team to supply a waveform package.');
    return out;
  }
  root.ECGArchive={files};
})(typeof globalThis!=='undefined'?globalThis:window);
