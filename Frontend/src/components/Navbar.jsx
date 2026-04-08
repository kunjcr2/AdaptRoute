import React from 'react';
import { NavLink } from 'react-router-dom';
import { Network } from 'lucide-react';
import { cn } from '../lib/utils';
import { motion } from 'framer-motion';

const Navbar = () => {
  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="sticky top-0 z-50 w-full border-b border-brand-200/50 bg-white/70 backdrop-blur-md"
    >
      <div className="container mx-auto px-6 h-16 flex items-center justify-between">
        <NavLink to="/" className="flex items-center gap-2 group">
          <div className="bg-brand-50 rounded-lg p-1.5 border border-brand-100 group-hover:bg-brand-100 transition-colors">
            <Network className="w-5 h-5 text-brand-700" />
          </div>
          <span className="font-serif text-xl font-semibold tracking-tight text-brand-900">AdaptRoute</span>
        </NavLink>

        <nav className="hidden md:flex items-center gap-8">
          {[
            { name: 'Home', path: '/' },
            { name: 'Architecture', path: '/architecture' },
            { name: 'Firewall', path: '/firewall' },
            { name: 'Evaluation', path: '/evaluation' }
          ].map((item) => (
            <NavLink
              key={item.name}
              to={item.path}
              className={({ isActive }) =>
                cn(
                  "text-sm font-medium transition-colors hover:text-brand-900",
                  isActive ? "text-brand-900 before:absolute before:-bottom-5 before:h-0.5 before:w-full before:bg-brand-900 relative" : "text-brand-600"
                )
              }
            >
              {item.name}
            </NavLink>
          ))}
        </nav>

        <a
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2 bg-brand-900 text-white px-4 py-2 rounded-full text-sm font-medium hover:bg-brand-800 transition-all shadow-sm shadow-brand-900/20"
        >
          {/* <Github className="w-4 h-4" /> */}
          <span>GitHub</span>
        </a>
      </div>
    </motion.header>
  );
};

export default Navbar;
