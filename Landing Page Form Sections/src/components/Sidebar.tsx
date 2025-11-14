import { Section } from "../App";
import { motion } from "motion/react";
import { Button } from "./ui/button";
import { HelpCircle } from "lucide-react";

interface SidebarProps {
  activeSection: Section;
  onSectionChange: (section: Section) => void;
}

const sections = [
  { id: "what-use" as Section, label: "What Use?" },
  { id: "brand-info" as Section, label: "Brand Info" },
  { id: "score" as Section, label: "Score" },
];

export function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  return (
    <aside className="w-64 border-r border-indigo-200 bg-white flex flex-col shadow-lg h-full">
      {sections.map((section) => (
        <motion.button
          key={section.id}
          onClick={() => onSectionChange(section.id)}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          style={{
            backgroundColor:
              activeSection === section.id ? "rgb(199, 186, 233)" : "white",
          }}
          className="border-b border-indigo-100 p-8 text-center transition-all duration-300 relative hover:bg-[rgb(199,186,233)]"
        >
          {activeSection === section.id && (
            <motion.div
              layoutId="activeSection"
              style={{ backgroundColor: "rgb(199, 186, 233)" }}
              className="absolute inset-0"
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
            />
          )}
          <div className="relative p-6 transition-all duration-300">
            {section.label}
          </div>
        </motion.button>
      ))}

      {/* Need Help Button */}
      <div className="mt-auto p-6">
        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Button className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white border-0 shadow-md gap-2">
            <HelpCircle className="w-4 h-4" />
            Need Help?
          </Button>
        </motion.div>
      </div>
    </aside>
  );
}
