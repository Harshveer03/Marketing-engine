import { useState } from "react";
import { motion } from "motion/react";
import {
  Plus,
  Home,
  Activity,
  Megaphone,
  FileText,
  BarChart3,
  Settings,
  Crown,
  User,
  Zap,
} from "lucide-react";
import { Button } from "../ui/button";

interface DashboardSidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

const menuItems = [
  { id: "new", label: "New", icon: Plus },
  { id: "home", label: "Home", icon: Home },
  { id: "diagnostics", label: "Diagnostics", icon: Activity },
  { id: "campaigns", label: "Campaigns", icon: Megaphone },
  { id: "posts", label: "Posts", icon: FileText },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
  { id: "profile", label: "Profile", icon: User },
  { id: "settings", label: "Settings", icon: Settings },
];

export function DashboardSidebar({
  activeSection,
  onSectionChange,
}: DashboardSidebarProps) {
  const [showNewMenu, setShowNewMenu] = useState(false);

  return (
    <aside className="w-40 bg-white border-r border-gray-200 flex flex-col">
      {/* Menu Items */}
      <nav className="flex-1 p-2 space-y-1">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeSection === item.id;

          // Special handling for "New" button with popup
          if (item.id === "new") {
            return (
              <div key={item.id} className="relative">
                <motion.button
                  onMouseEnter={() => setShowNewMenu(true)}
                  onMouseLeave={() => setShowNewMenu(false)}
                  onClick={() => setShowNewMenu(!showNewMenu)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  style={{
                    backgroundColor: isActive ? "#000000" : "transparent",
                  }}
                  className={`w-full flex flex-col items-center gap-1 py-3 px-2 rounded-xl transition-all ${
                    isActive
                      ? "text-white shadow-lg"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="text-xs font-medium">{item.label}</span>
                </motion.button>

                {/* Popup Menu */}
                {showNewMenu && (
                  <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.2 }}
                    onMouseEnter={() => setShowNewMenu(true)}
                    onMouseLeave={() => setShowNewMenu(false)}
                    className="absolute left-full top-0 ml-2 bg-white border-2 border-gray-200 rounded-xl shadow-xl z-50 overflow-hidden"
                    style={{ minWidth: "180px" }}
                  >
                    {/* New Post */}
                    <motion.button
                      whileHover={{ backgroundColor: "#f3f4f6" }}
                      onClick={() => {
                        onSectionChange("posts");
                        setShowNewMenu(false);
                      }}
                      className="w-full flex items-center gap-3 px-4 py-3 text-left text-gray-700 hover:text-gray-900 transition-colors"
                    >
                      <FileText className="w-5 h-5" />
                      <span className="text-sm font-medium">New Post</span>
                    </motion.button>

                    {/* Divider */}
                    <div className="border-t border-gray-200" />

                    {/* New Campaign */}
                    <motion.button
                      whileHover={{ backgroundColor: "#f3f4f6" }}
                      onClick={() => {
                        onSectionChange("campaigns");
                        setShowNewMenu(false);
                      }}
                      className="w-full flex items-center gap-3 px-4 py-3 text-left text-gray-700 hover:text-gray-900 transition-colors"
                    >
                      <Zap className="w-5 h-5" />
                      <span className="text-sm font-medium">New Campaign</span>
                    </motion.button>
                  </motion.div>
                )}
              </div>
            );
          }

          // Regular menu items
          return (
            <motion.button
              key={item.id}
              onClick={() => onSectionChange(item.id)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              style={{
                backgroundColor: isActive ? "#000000" : "transparent",
              }}
              className={`w-full flex flex-col items-center gap-1 py-3 px-2 rounded-xl transition-all ${
                isActive
                  ? "text-white shadow-lg"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="text-xs font-medium">{item.label}</span>
            </motion.button>
          );
        })}
      </nav>

      {/* Upgrade Button */}
      <div className="p-2">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="w-full bg-gradient-to-br from-gray-900 to-gray-700 py-3 px-2 rounded-xl text-yellow-400 flex flex-col items-center gap-1"
        >
          <Crown className="w-5 h-5" />
          <span className="text-xs font-medium text-white">Upgrade</span>
        </motion.button>
      </div>
    </aside>
  );
}
