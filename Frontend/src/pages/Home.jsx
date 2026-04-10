import React from 'react';
import { motion } from 'framer-motion';
import { Cpu, Combine, Shield, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

const Home = () => {
  return (
    <div className="w-full">
      {/* Hero Section */}
      <section className="container mx-auto px-6 pt-24 pb-32 text-center max-w-5xl">
        <motion.div {...fadeIn}>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-brand-200 text-brand-800 text-xs font-semibold uppercase tracking-wider mb-8 shadow-sm">
            <span className="w-2 h-2 rounded-full bg-brand-500 animate-pulse"></span>
            Edge-Optimized Inference
          </div>
          <h1 className="font-serif text-4xl md:text-5xl font-bold leading-tight mb-8 text-brand-950">
            Task-Aware SLM Routing
            <span className="block text-brand-500 italic font-medium mt-2">with Soft LoRA Merging</span>
          </h1>
          <p className="text-xl text-brand-700 leading-relaxed max-w-3xl mx-auto mb-12 font-light">
            A learned gating network that dynamically blends LoRA expert adapters at inference time.
            Sparse MoE-style routing without the end-to-end training cost, built for devices where small models must punch above their weight.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/architecture" className="flex items-center justify-center gap-2 bg-brand-900 text-white px-8 py-4 rounded-full font-medium hover:bg-brand-800 transition-all shadow-xl hover:shadow-2xl hover:-translate-y-0.5">
              Explore the Architecture
              <ArrowRight className="w-4 h-4" />
            </Link>
            <a href="https://github.com/kunjcr2/AdaptRoute" className="bg-white/80 backdrop-blur-sm text-brand-900 border border-brand-200 px-8 py-4 rounded-full font-medium hover:bg-white transition-all shadow-sm">
              Read More
            </a>
          </div>
        </motion.div>
      </section>

      {/* Features Showcase Section */}
      <section className="bg-white/60 backdrop-blur-xl border-y border-brand-200/50">
        <div className="container mx-auto px-6 py-24">
          <div className="max-w-3xl mx-auto text-center mb-16">
            <h2 className="font-serif text-4xl font-bold mb-6 text-brand-900">Why AdaptRoute?</h2>
            <p className="text-lg text-brand-600 font-light leading-relaxed">
              Small language models (SLMs) are essential for edge deployment, but they natively struggle as generalists. <strong>They are simply not powerful enough to excel simultaneously at vastly different tasks like writing code, solving math, medical triage, and answering domain-specific queries.</strong>
              <br /><br />
              AdaptRoute solves this. By dynamically attaching lightweight LoRA adapters at inference time, it gives these models the specialized expertise they need exactly when they need it—all with <em>minimal latency overhead</em>.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {[
              {
                icon: <Cpu className="w-6 h-6" />,
                title: "Edge-First Design",
                desc: "One frozen base model, multiple lightweight LoRA adapters (~40MB each). Fits easily on mid-range mobile devices with no cloud dependency.",
                link: "/architecture"
              },
              {
                icon: <Combine className="w-6 h-6" />,
                title: "Soft Routing",
                desc: "Blends top-k adapters by probability using PEFT's additive weighting, gracefully handling ambiguous queries needing cross-domain expertise.",
                link: "/architecture"
              },
              {
                icon: <Shield className="w-6 h-6" />,
                title: "Injection Firewall",
                desc: "The DistilBERT gate pre-filters queries, blocking prompt injections before generation occurs. Critical for systems with local tool access.",
                link: "/firewall"
              }
            ].map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="bg-white p-10 rounded-3xl border border-brand-100 shadow-xl shadow-brand-200/20 hover:shadow-2xl hover:shadow-brand-300/30 transition-all flex flex-col group"
              >
                <div className="w-14 h-14 rounded-2xl bg-brand-50 flex items-center justify-center text-brand-800 mb-8 shadow-inner group-hover:scale-110 transition-transform">
                  {feature.icon}
                </div>
                <h3 className="text-2xl font-bold mb-4 text-brand-900">{feature.title}</h3>
                <p className="text-brand-600 leading-relaxed text-sm flex-1 mb-8">{feature.desc}</p>
                <Link to={feature.link} className="inline-flex items-center text-sm font-semibold text-brand-700 hover:text-brand-950">
                  Learn more <ArrowRight className="w-4 h-4 ml-1" />
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
