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
  { id: "diagnostics", label: "Brand Diagnostics", icon: Activity },
  { id: "campaigns", label: "Campaigns", icon: Megaphone },
  { id: "posts", label: "Your Posts", icon: FileText },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
  { id: "calendar", label: "Calendar", icon: Calendar },
  { id: "settings", label: "Settings", icon: Settings },
];

export function DashboardSidebar({
  activeSection,
  onSectionChange,
}: DashboardSidebarProps) {
  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-gray-200">
        <h1 className="text-2xl font-bold text-gray-900">1SYX</h1>
      </div>

      {/* Menu Items */}
      <nav className="flex-1 p-4 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeSection === item.id;

          return (
            <motion.button
              key={item.id}
              onClick={() => onSectionChange(item.id)}
              whileHover={{ x: 4 }}
              whileTap={{ scale: 0.98 }}
              style={{
                backgroundColor: isActive ? "#000000" : "transparent",
              }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                isActive
                  ? "text-white shadow-lg"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </motion.button>
          );
        })}
      </nav>

      {/* Upgrade Card */}
      <div className="p-4">
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-gradient-to-br from-gray-900 to-gray-700 p-6 rounded-2xl text-white"
        >
          <Crown className="w-8 h-8 mb-3 text-yellow-400" />
          <h3 className="font-bold text-lg mb-2">Upgrade to Pro</h3>
          <p className="text-sm text-gray-300 mb-4">
            Unlock unlimited features and credits
          </p>
          <Button className="w-full bg-white text-black hover:bg-gray-100 font-semibold">
            Upgrade Now
          </Button>
        </motion.div>
      </div>
    </aside>
  );
}
