import React from 'react';
import { motion } from 'framer-motion';
import { GitMerge, BrainCircuit, Blocks } from 'lucide-react';

const Architecture = () => {
  return (
    <div className="container mx-auto px-6 py-20 max-w-5xl">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-16 text-center"
      >
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white shadow-xl mb-6 border border-brand-100 text-brand-800">
           <Blocks className="w-8 h-8" />
        </div>
        <h1 className="font-serif text-5xl font-bold mb-6 text-brand-900">System Architecture</h1>
        <p className="text-xl text-brand-600 font-light max-w-2xl mx-auto">
          A decoupled approach to MoE routing that makes specialized inference practical for the edge.
        </p>
      </motion.div>

      {/* Visual Pipeline */}
      <div className="bg-white/80 backdrop-blur-xl p-12 rounded-[2.5rem] shadow-2xl border border-brand-100 relative overflow-hidden mb-24">
         <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-brand-400 to-transparent"></div>
         
         <div className="flex flex-col items-center gap-8 relative z-10 w-full max-w-2xl mx-auto">
            {/* Input */}
            <div className="bg-brand-950 text-white px-8 py-4 rounded-2xl font-medium shadow-xl w-64 text-center">
              User Query
            </div>
            
            <div className="h-10 w-px bg-brand-300"></div>

            {/* Gate */}
            <div className="bg-white border-2 border-brand-200 w-full p-8 rounded-3xl text-center shadow-lg relative group hover:border-brand-400 transition-all">
              <div className="flex items-center justify-center gap-3 mb-4">
                <BrainCircuit className="w-6 h-6 text-brand-600" />
                <div className="font-bold text-xl text-brand-900">Gating Network (DistilBERT)</div>
              </div>
              <p className="text-brand-600 text-sm mb-6">66M parameters • ~5ms cpu inference</p>
              
              <div className="flex justify-center gap-3 mb-6">
                 {['p(code)', 'p(math)', 'p(QA)', 'p(medical)'].map(t => (
                   <span key={t} className="bg-brand-50 text-brand-700 font-mono text-xs px-3 py-1.5 rounded-lg border border-brand-100">{t}</span>
                 ))}
              </div>
            </div>

            <div className="h-10 w-px bg-brand-300"></div>

            {/* Merge */}
            <div className="bg-gradient-to-r from-brand-50 via-white to-brand-50 border border-brand-200 w-full p-6 rounded-2xl text-center shadow-inner flex flex-col items-center gap-3">
              <GitMerge className="w-6 h-6 text-brand-600" />
              <span className="font-mono text-brand-800 font-medium">Soft Merge: add_weighted_adapter(adapters, weights)</span>
            </div>

            <div className="h-10 w-px bg-brand-300"></div>

            {/* Base */}
            <div className="bg-brand-800 text-white w-full p-8 rounded-3xl text-center shadow-xl relative overflow-hidden">
               <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
               <div className="relative z-10">
                <div className="font-bold text-2xl mb-2">Base SLM (Qwen2.5-1.5B)</div>
                <div className="text-brand-200 font-medium bg-white/10 inline-block px-4 py-1.5 rounded-full mt-2">
                  + Merged LoRA Experts
                </div>
               </div>
            </div>
         </div>
      </div>

       {/* Detailed Text */}
       <div className="grid md:grid-cols-2 gap-16">
          <div>
            <h3 className="font-serif text-3xl font-bold mb-6 text-brand-900">The Base Model</h3>
            <p className="text-brand-600 leading-relaxed mb-6">
              We utilize Qwen2.5-1.5B loaded in 4-bit NF4 quantization via `bitsandbytes`. It requires less than 4GB VRAM, making it the perfect foundation. The weights remain fully frozen.
            </p>
            <h3 className="font-serif text-3xl font-bold mb-6 mt-12 text-brand-900">Expert Adapters</h3>
            <p className="text-brand-600 leading-relaxed">
              Four domain-specific adapters (`lora-code`, `lora-math`, `lora-qa`, `lora-medical`) trained via SFT. At rank 8, each adapter is incredibly light—costing less than 200MB to store all four domains concurrently.
            </p>
          </div>
          <div className="bg-white p-8 rounded-3xl border border-brand-100 shadow-lg">
             <h3 className="font-serif text-2xl font-bold mb-6 text-brand-900">Soft Routing Mechanics</h3>
             <div className="bg-[#1e1e1e] rounded-xl p-6 overflow-x-auto text-sm font-mono text-gray-300 shadow-inner">
<pre>{`probs = gate_model(query)

top_adapters = ["lora-code", "lora-math"]
top_weights  = [0.72, 0.21]

model.add_weighted_adapter(
    adapters=top_adapters,
    weights=top_weights,
    adapter_name="merged"
)

response = model.generate(query)`}</pre>
             </div>
             <p className="mt-6 text-brand-600 text-sm">
                Single forward pass with blended expertise.
             </p>
          </div>
       </div>
    </div>
  );
};

export default Architecture;
