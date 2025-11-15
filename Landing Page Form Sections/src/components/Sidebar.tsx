import { Section } from "../App";
import { motion } from "motion/react";
import { Button } from "./ui/button";
import { HelpCircle, FileText, Eye, BarChart3, Target } from "lucide-react";

interface SidebarProps {
  activeSection: Section;
  onSectionChange: (section: Section) => void;
}

const sections = [
  { id: "brand-info" as Section, label: "Brand Info", icon: FileText, color: "from-blue-500 to-cyan-500" },
  { id: "preview" as Section, label: "Preview", icon: Eye, color: "from-purple-500 to-pink-500" },
  { id: "score" as Section, label: "Score", icon: BarChart3, color: "from-green-500 to-emerald-500" },
  { id: "what-use" as Section, label: "What Use?", icon: Target, color: "from-orange-500 to-red-500" },
];

export function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  return (
    <aside className="w-72 border-r border-indigo-200 bg-gradient-to-b from-white to-indigo-50/30 flex flex-col shadow-xl min-h-full">
      <div className="p-6 border-b border-indigo-100">
        <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider">Navigation</h3>
      </div>
      
      {sections.map((section, index) => {
        const Icon = section.icon;
        const isActive = activeSection === section.id;
        
        return (
          <motion.button
            key={section.id}
            onClick={() => onSectionChange(section.id)}
            whileHover={{ x: 4 }}
            whileTap={{ scale: 0.98 }}
            className={`border-b border-indigo-100 p-6 text-left transition-all duration-300 relative group ${
              isActive ? "bg-gradient-to-r from-indigo-50 to-purple-50" : "hover:bg-indigo-50/50"
            }`}
          >
            {isActive && (
              <motion.div
                layoutId="activeSection"
                className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-indigo-500 to-purple-500"
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
              />
            )}
            
            <div className="relative flex items-center gap-4">
              <motion.div 
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                  isActive 
                    ? `bg-gradient-to-br ${section.color} shadow-lg` 
                    : "bg-gray-100 group-hover:bg-gradient-to-br group-hover:" + section.color
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? "text-white" : "text-gray-600 group-hover:text-white"}`} />
              </motion.div>
              
              <div className="flex-1">
                <div className={`font-semibold transition-colors ${
                  isActive ? "text-indigo-700" : "text-gray-700 group-hover:text-indigo-600"
                }`}>
                  {section.label}
                </div>
                <div className="text-xs text-gray-500 mt-0.5">
                  {index === 0 && "Enter details"}
                  {index === 1 && "Review info"}
                  {index === 2 && "View results"}
                  {index === 3 && "Select option"}
                </div>
              </div>
              
              {isActive && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="w-2 h-2 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500"
                />
              )}
            </div>
          </motion.button>
        );
      })}

      {/* Need Help Button */}
      <div className="mt-auto p-6 border-t border-indigo-100 bg-gradient-to-br from-indigo-50 to-purple-50">
        <motion.div whileHover={{ scale: 1.03, y: -2 }} whileTap={{ scale: 0.97 }}>
          <Button className="w-full bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-700 hover:via-purple-700 hover:to-pink-700 text-white border-0 shadow-lg hover:shadow-xl gap-2 py-6 rounded-xl transition-all duration-300">
            <HelpCircle className="w-5 h-5" />
            <span className="font-semibold">Need Help?</span>
          </Button>
        </motion.div>
        <p className="text-xs text-center text-gray-500 mt-3">
          We're here to assist you 24/7
        </p>
      </div>
    </aside>
  );
}
