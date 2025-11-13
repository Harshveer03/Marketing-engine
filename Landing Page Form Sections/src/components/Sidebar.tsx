import { Section } from '../App';
import { motion } from 'motion/react';

interface SidebarProps {
  activeSection: Section;
  onSectionChange: (section: Section) => void;
}

const sections = [
  { id: 'what-use' as Section, label: 'What Use?' },
  { id: 'brand-info' as Section, label: 'Brand Info' },
  { id: 'score' as Section, label: 'Score' },
];

export function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  return (
    <aside className="w-64 border-r border-indigo-200 bg-white/50 backdrop-blur-sm flex flex-col shadow-lg">
      {sections.map((section, index) => (
        <motion.button
          key={section.id}
          onClick={() => onSectionChange(section.id)}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className={`border-b border-indigo-100 p-8 text-center transition-all duration-300 relative ${
            activeSection === section.id
              ? 'bg-white shadow-inner'
              : 'bg-transparent hover:bg-white/50'
          }`}
        >
          {activeSection === section.id && (
            <motion.div
              layoutId="activeSection"
              className="absolute inset-0 bg-gradient-to-r from-indigo-100 to-purple-100"
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
            />
          )}
          <div className={`relative border-2 rounded-lg p-6 transition-all duration-300 ${
            activeSection === section.id
              ? 'border-indigo-500 bg-gradient-to-br from-indigo-500 to-purple-500 text-white shadow-lg'
              : 'border-indigo-200 bg-white hover:border-indigo-400'
          }`}>
            {section.label}
          </div>
        </motion.button>
      ))}
    </aside>
  );
}