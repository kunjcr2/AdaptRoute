import React from 'react';
import { motion } from 'framer-motion';
import { BarChart, CheckCircle2, Target } from 'lucide-react';

const Evaluation = () => {
  return (
    <div className="container mx-auto px-6 py-20 max-w-5xl">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-16 text-center"
      >
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white shadow-xl mb-6 border border-brand-100 text-brand-800">
           <BarChart className="w-8 h-8" />
        </div>
        <h1 className="font-serif text-5xl font-bold mb-6 text-brand-900">Evaluation Metrics</h1>
        <p className="text-xl text-brand-600 font-light max-w-2xl mx-auto">
          Measuring the effectiveness of the DistilBERT gating network and the resulting quality delta from soft routing.
        </p>
      </motion.div>

      <div className="grid md:grid-cols-2 gap-8 mb-16">
         {/* Performance Targets */}
         <div className="bg-white p-8 rounded-3xl border border-brand-100 shadow-lg">
            <h3 className="font-serif text-2xl font-bold mb-6 text-brand-900 flex items-center gap-3">
              <Target className="w-6 h-6 text-brand-500" /> Key Targets
            </h3>
            <ul className="space-y-4">
               {[
                 { label: "Gate Accuracy", desc: "> 90% routing to correct adapter." },
                 { label: "Routing Latency", desc: "< 10ms for Gate inference." },
                 { label: "Merge Time", desc: "< 100ms for add_weighted_adapter()." },
                 { label: "Injection FPR", desc: "< 2% legitimate requests blocked." },
                 { label: "Injection Recall", desc: "> 95% of attack attempts caught." }
               ].map((item, i) => (
                 <li key={i} className="flex gap-4">
                   <CheckCircle2 className="w-5 h-5 text-green-500 shrink-0 mt-0.5" />
                   <div>
                     <span className="font-bold text-brand-900 block">{item.label}</span>
                     <span className="text-sm text-brand-600">{item.desc}</span>
                   </div>
                 </li>
               ))}
            </ul>
         </div>

         {/* Training Data */}
         <div className="bg-white p-8 rounded-3xl border border-brand-100 shadow-lg">
            <h3 className="font-serif text-2xl font-bold mb-6 text-brand-900 flex items-center gap-3">
              <BarChart className="w-6 h-6 text-brand-500" /> Datasets
            </h3>
            <div className="space-y-6">
              <div>
                <h4 className="font-bold text-brand-800 mb-2">Gate Training</h4>
                <div className="flex flex-wrap gap-2 text-sm">
                  <span className="px-3 py-1 bg-brand-50 border border-brand-200 rounded-full">ought/raft</span>
                  <span className="px-3 py-1 bg-brand-50 border border-brand-200 rounded-full">github-code</span>
                  <span className="px-3 py-1 bg-brand-50 border border-brand-200 rounded-full">MATH</span>
                  <span className="px-3 py-1 bg-brand-50 border border-brand-200 rounded-full">squad</span>
                  <span className="px-3 py-1 bg-brand-50 border border-brand-200 rounded-full">prompt-injections</span>
                </div>
              </div>
              <div>
                <h4 className="font-bold text-brand-800 mb-2">Adapter SFT</h4>
                <div className="flex flex-wrap gap-2 text-sm">
                  <span className="px-3 py-1 bg-brand-50 border border-brand-200 rounded-full">python_code_instructions</span>
                  <span className="px-3 py-1 bg-brand-50 border border-brand-200 rounded-full">MATH</span>
                  <span className="px-3 py-1 bg-brand-50 border border-brand-200 rounded-full">squad_v2</span>
                  <span className="px-3 py-1 bg-brand-50 border border-brand-200 rounded-full">cnn_dailymail</span>
                </div>
              </div>
            </div>
         </div>
      </div>
      
      <div className="bg-brand-900 text-white p-10 rounded-3xl shadow-xl text-center">
         <h2 className="font-serif text-3xl font-bold mb-4">The Demo</h2>
         <p className="text-brand-200 max-w-3xl mx-auto">
            The key evaluation is a side-by-side comparison: the exact same query answered by the base model alone versus the soft-routed adapter blend. The difference in response specificity and accuracy constitutes the "quality delta".
         </p>
      </div>
    </div>
  );
};

export default Evaluation;
