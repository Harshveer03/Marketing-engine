import { Section } from "../App";
import { motion, AnimatePresence } from "motion/react";
import { Button } from "./ui/button";
import {
  HelpCircle,
  FileText,
  Eye,
  BarChart3,
  Target,
  Menu,
} from "lucide-react";
import { useState } from "react";

interface SidebarProps {
  activeSection: Section;
  onSectionChange: (section: Section) => void;
}

const sections = [
  {
    id: "brand-info" as Section,
    label: "Brand Info",
    icon: FileText,
    description: "Enter details",
  },
  {
    id: "preview" as Section,
    label: "Preview",
    icon: Eye,
    description: "Review info",
  },
  {
    id: "score" as Section,
    label: "Score",
    icon: BarChart3,
    description: "View results",
  },
  {
    id: "what-use" as Section,
    label: "What Use?",
    icon: Target,
    description: "Select option",
  },
];

export function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <motion.aside
      animate={{ width: isCollapsed ? 80 : 288 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="border-r border-gray-200 bg-white flex flex-col shadow-xl min-h-full"
    >
      <div className="p-6 border-b border-gray-200 flex items-center justify-center">
        <motion.button
          onClick={() => setIsCollapsed(!isCollapsed)}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <Menu className="w-6 h-6 text-gray-900" />
        </motion.button>
      </div>

      {sections.map((section) => {
        const Icon = section.icon;
        const isActive = activeSection === section.id;

        return (
          <motion.button
            key={section.id}
            onClick={() => onSectionChange(section.id)}
            whileHover={{ x: isCollapsed ? 0 : 4 }}
            whileTap={{ scale: 0.98 }}
            className={`border-b border-gray-200 p-6 text-left transition-all duration-300 relative group ${
              isActive ? "bg-gray-100" : "hover:bg-gray-50"
            }`}
            title={isCollapsed ? section.label : ""}
          >
            {isActive && (
              <motion.div
                layoutId="activeSection"
                className="absolute left-0 top-0 bottom-0 w-1 bg-black"
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
              />
            )}

            <div
              className={`relative flex items-center ${
                isCollapsed ? "justify-center" : "gap-4"
              }`}
            >
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                  isActive
                    ? "bg-gray-100 shadow-lg"
                    : "bg-gray-100 group-hover:bg-gray-200"
                }`}
              >
                <Icon className="w-5 h-5 text-black" />
              </motion.div>

              <AnimatePresence>
                {!isCollapsed && (
                  <motion.div
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: "auto" }}
                    exit={{ opacity: 0, width: 0 }}
                    transition={{ duration: 0.2 }}
                    className="flex-1 overflow-hidden"
                  >
                    <div
                      className={`font-semibold transition-colors whitespace-nowrap ${
                        isActive
                          ? "text-black"
                          : "text-gray-700 group-hover:text-black"
                      }`}
                    >
                      {section.label}
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5 whitespace-nowrap">
                      {section.description}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {isActive && !isCollapsed && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="w-2 h-2 rounded-full bg-black"
                />
              )}
            </div>
          </motion.button>
        );
      })}

      {/* Need Help Button */}
      <div className="mt-auto p-6 border-t border-gray-200 bg-gray-50">
        <motion.div
          whileHover={{ scale: 1.03, y: -2 }}
          whileTap={{ scale: 0.97 }}
        >
          <Button
            className={`w-full !bg-black hover:bg-gray-800 text-white border-0 shadow-lg hover:shadow-xl py-6 rounded-xl transition-all duration-300 ${
              isCollapsed ? "px-0 justify-center" : "gap-2"
            }`}
            title={isCollapsed ? "Need Help?" : ""}
          >
            <HelpCircle className="w-5 h-5" />
            <AnimatePresence>
              {!isCollapsed && (
                <motion.span
                  initial={{ opacity: 0, width: 0 }}
                  animate={{ opacity: 1, width: "auto" }}
                  exit={{ opacity: 0, width: 0 }}
                  transition={{ duration: 0.2 }}
                  className="font-semibold whitespace-nowrap overflow-hidden"
                >
                  Need Help?
                </motion.span>
              )}
            </AnimatePresence>
          </Button>
        </motion.div>
        <AnimatePresence>
          {!isCollapsed && (
            <motion.p
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="text-xs text-center text-gray-500 mt-3 overflow-hidden"
            >
              We're here to assist you 24/7
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </motion.aside>
  );
}
