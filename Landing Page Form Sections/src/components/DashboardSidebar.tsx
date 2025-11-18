import { motion } from "motion/react";
import {
  Plus,
  Home,
  Activity,
  Megaphone,
  FileText,
  BarChart3,
  Calendar,
  Settings,
  Crown,
} from "lucide-react";
import { Button } from "./ui/button";

interface DashboardSidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

const menuItems = [
  { id: "new", label: "+ New", icon: Plus },
  { id: "home", label: "Home", icon: Home },
  { id: "diagnostics", label: "Diagnostics", icon: Activity },
  { id: "campaigns", label: "Campaigns", icon: Megaphone },
  { id: "posts", label: "Posts", icon: FileText },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
  { id: "calendar", label: "Calendar", icon: Calendar },
  { id: "settings", label: "Settings", icon: Settings },
];

export function DashboardSidebar({
  activeSection,
  onSectionChange,
}: DashboardSidebarProps) {
  return (
    <aside className="w-40 bg-white border-r border-gray-200 flex flex-col">
      {/* Menu Items */}
      <nav className="flex-1 p-2 space-y-1">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeSection === item.id;

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
