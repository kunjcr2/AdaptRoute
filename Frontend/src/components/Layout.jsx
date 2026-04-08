import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import { Network } from 'lucide-react';

const Layout = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Outlet />
      </main>
      
      <footer className="mt-auto border-t border-brand-200/50 bg-white/50 py-12 backdrop-blur-sm">
        <div className="container mx-auto px-6 text-center">
          <div className="font-serif text-xl font-bold text-brand-900 mb-4 flex items-center justify-center gap-2">
            <Network className="w-5 h-5 text-brand-600" /> AdaptRoute
          </div>
          <p className="text-brand-600 text-sm max-w-md mx-auto">
            A learned gating network that dynamically blends LoRA expert adapters at inference time. Edge-optimized system.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
