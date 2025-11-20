import { Section } from "../../App";
import { motion } from "motion/react";
import { Button } from "../ui/button";
import { HelpCircle, FileText, Eye, Target } from "lucide-react";

interface SidebarProps {
  activeSection: Section;
  onSectionChange: (section: Section) => void;
}

const sections = [
  {
    id: "brand-info" as Section,
    label: "Brand Info",
    icon: FileText,
  },
  {
    id: "preview" as Section,
    label: "Preview",
    icon: Eye,
  },
  {
    id: "what-use" as Section,
    label: "What Use?",
    icon: Target,
  },
];

export function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  return (
    <aside className="w-20 border-r border-gray-200 bg-white/90 backdrop-blur-sm flex flex-col shadow-xl min-h-full">
      {sections.map((section) => {
        const Icon = section.icon;
        const isActive = activeSection === section.id;

        return (
          <motion.button
            key={section.id}
            onClick={() => onSectionChange(section.id)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            style={{
              backgroundColor: isActive ? "#ffffffff" : "transparent",
            }}
            className={`border-b border-gray-200 p-6 transition-all duration-300 relative group ${
              !isActive && "hover:bg-gray-50"
            }`}
            title={section.label}
          >
            {isActive && (
              <motion.div
                layoutId="activeSection"
                className="absolute left-0 top-0 bottom-0 w-1 bg-white"
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
              />
            )}

            <div className="relative flex items-center justify-center">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                  isActive
                    ? "bg-white shadow-lg"
                    : "bg-gray-100 group-hover:bg-gray-200"
                }`}
              >
                <Icon
                  className={`w-5 h-5 ${
                    isActive ? "text-black" : "text-black"
                  }`}
                />
              </motion.div>
            </div>
          </motion.button>
        );
      })}

      {/* Need Help Button */}
      <div className="mt-auto p-6 border-t border-gray-200 bg-gray-50">
        <motion.div
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
        >
          <Button
            className="w-full !bg-black hover:!bg-gray-800 text-white border-0 shadow-lg hover:shadow-xl p-3 rounded-xl transition-all duration-300 flex items-center justify-center"
            title="Need Help?"
          >
            <HelpCircle className="w-5 h-5" />
          </Button>
        </motion.div>
      </div>
    </aside>
  );
}
