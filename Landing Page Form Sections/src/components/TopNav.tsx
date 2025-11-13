import { Section } from "../App";
import { Button } from "./ui/button";
import { ChevronDown, HelpCircle, Edit2 } from "lucide-react";
import { motion } from "motion/react";

interface TopNavProps {
  activeSection: Section;
}

const sectionTitles = {
  "what-use": "Brand Discovery Form",
  "brand-info": "Brand Discovery Form",
  resources: "Brand Discovery Form",
  score: "Brand Discovery Form",
};

const tabs = ["Summary", "Previews", "Report"];

export function TopNav({ activeSection }: TopNavProps) {
  return (
    <div className="bg-white/95 backdrop-blur-md border-b border-indigo-100 shadow-sm sticky top-[57px] z-40">
      <div className="px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-lg text-gray-800">
            {sectionTitles[activeSection]}
          </h1>
          <button className="p-1.5 hover:bg-indigo-50 rounded-lg transition-all duration-200">
            <Edit2 className="w-4 h-4 text-gray-500" />
          </button>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            className="gap-2 border-indigo-200 text-gray-700 hover:bg-indigo-50 hover:border-indigo-300 rounded-lg transition-all duration-200"
          >
            yourapp.com <ChevronDown className="w-4 h-4" />
          </Button>

          <Button
            variant="outline"
            size="sm"
            className="gap-2 border-indigo-200 text-gray-700 hover:bg-indigo-50 hover:border-indigo-300 rounded-lg transition-all duration-200"
          >
            US <ChevronDown className="w-4 h-4" />
          </Button>

          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button
              size="sm"
              className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white border-0 shadow-md rounded-lg gap-2 relative overflow-hidden"
            >
              <div
                className="absolute inset-0 opacity-20 bg-cover bg-center"
                style={{
                  backgroundImage: `url('https://images.unsplash.com/photo-1646038572891-86b08ccd6719?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwd2F2ZXN8ZW58MXx8fHwxNzYzMDExMDc0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
                }}
              />
              <span className="relative z-10 flex items-center gap-2">
                <HelpCircle className="w-4 h-4" />
                Need Help?
              </span>
            </Button>
          </motion.div>
        </div>
      </div>

      <div className="px-8 flex gap-8 border-t border-indigo-50">
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
    </div>
  );
}
