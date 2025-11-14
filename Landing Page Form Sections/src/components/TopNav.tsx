import { Section } from "../App";
import { Button } from "./ui/button";
import { ChevronDown, HelpCircle, Edit2, RotateCcw } from "lucide-react";
import { motion } from "motion/react";

interface TopNavProps {
  activeSection: Section;
  onClearForm: () => void;
}

const sectionTitles = {
  "what-use": "Brand Discovery Form",
  "brand-info": "Brand Discovery Form",
  resources: "Brand Discovery Form",
  score: "Brand Discovery Form",
};

const tabs = ["Summary", "Previews", "Report"];

export function TopNav({ activeSection, onClearForm }: TopNavProps) {
  return (
    <div className="bg-white/95 backdrop-blur-md border-b border-indigo-100 shadow-sm sticky top-[57px] z-40">
      <div className="px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-8">
          {tabs.map((tab, index) => (
            <motion.button
              key={tab}
              whileHover={{ y: -2 }}
              className={`px-1 py-3 border-b-2 transition-all duration-200 relative ${
                index === 0
                  ? "border-indigo-600 text-indigo-600"
                  : "border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300"
              }`}
            >
              {tab}
              {index === 0 && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute -bottom-[2px] left-0 right-0 h-0.5 bg-gradient-to-r from-indigo-600 to-purple-600"
                />
              )}
            </motion.button>
          ))}
        </div>

        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Button
            onClick={onClearForm}
            variant="outline"
            className="border-2 border-gray-300 hover:border-red-400 hover:bg-red-50 text-gray-700 hover:text-red-600 transition-all duration-200"
          >
            <span className="flex items-center gap-2">
              <RotateCcw className="w-4 h-4" />
              Clear Form
            </span>
          </Button>
        </motion.div>
      </div>
    </div>
  );
}
