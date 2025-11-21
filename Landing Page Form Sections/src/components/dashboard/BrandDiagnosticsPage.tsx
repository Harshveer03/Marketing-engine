import { useState } from "react";
import { motion } from "motion/react";
import {
  BarChart3,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Target,
  MessageSquare,
  Calendar,
  Activity,
} from "lucide-react";

type MainTab = "industry" | "icp" | "persona" | "geography";

interface FilterOptions {
  industry: string[];
  icp: string[];
  persona: string[];
  geography: string[];
}

const filterOptions: FilterOptions = {
  industry: [
    "Technology",
    "Healthcare",
    "Finance",
    "E-commerce",
    "Manufacturing",
    "Education",
  ],
  icp: ["Enterprise", "SMB", "Startups", "Government", "Non-Profit"],
  persona: [
    "CEO",
    "Marketing Manager",
    "Product Manager",
    "Developer",
    "Sales Director",
  ],
  geography: [
    "North America",
    "Europe",
    "Asia Pacific",
    "Latin America",
    "Middle East",
    "Africa",
  ],
};

export function BrandDiagnosticsPage() {
  const [activeTab, setActiveTab] = useState<MainTab>("industry");
  const [selectedFilter, setSelectedFilter] = useState<string>(
    filterOptions.industry[0]
  );

  // Update selected filter when tab changes
  const handleTabChange = (tab: MainTab) => {
    setActiveTab(tab);
    setSelectedFilter(filterOptions[tab][0]);
  };

  const tabs = [
    { id: "industry" as MainTab, label: "Industry" },
    { id: "icp" as MainTab, label: "ICP" },
    { id: "persona" as MainTab, label: "Persona" },
    { id: "geography" as MainTab, label: "Geography" },
  ];

  const diagnosticSections = [
    {
      id: "clarity",
      title: "Clarity",
      icon: Target,
      score: 85,
      description: "How clearly your brand communicates its value proposition",
      color: "blue",
    },
    {
      id: "specificity",
      title: "Specificity",
      icon: CheckCircle,
      score: 72,
      description: "How specific and concrete your brand claims are",
      color: "green",
    },
    {
      id: "relevance",
      title: "Relevance",
      icon: Activity,
      score: 88,
      description: "How well your content aligns with audience interests",
      color: "purple",
    },
    {
      id: "gap",
      title: "GAP",
      icon: AlertTriangle,
      score: 65,
      description: "Gaps between your positioning and competitor strengths",
      color: "red",
    },
    {
      id: "messaging",
      title: "Messaging Consistency",
      icon: MessageSquare,
      score: 90,
      description: "Consistency of messaging across all channels",
      color: "indigo",
    },
    {
      id: "posting",
      title: "Posting Consistency",
      icon: Calendar,
      score: 78,
      description: "Regularity and frequency of content posting",
      color: "orange",
    },
    {
      id: "trend",
      title: "Trend",
      icon: TrendingUp,
      score: 82,
      description: "Performance trends and growth patterns over time",
      color: "pink",
    },
  ];

  const getColorClasses = (color: string) => {
    const colors: Record<
      string,
      { bg: string; text: string; border: string; progress: string }
    > = {
      blue: {
        bg: "bg-blue-50",
        text: "text-blue-600",
        border: "border-blue-200",
        progress: "bg-blue-500",
      },
      green: {
        bg: "bg-green-50",
        text: "text-green-600",
        border: "border-green-200",
        progress: "bg-green-500",
      },
      purple: {
        bg: "bg-purple-50",
        text: "text-purple-600",
        border: "border-purple-200",
        progress: "bg-purple-500",
      },
      red: {
        bg: "bg-red-50",
        text: "text-red-600",
        border: "border-red-200",
        progress: "bg-red-500",
      },
      indigo: {
        bg: "bg-indigo-50",
        text: "text-indigo-600",
        border: "border-indigo-200",
        progress: "bg-indigo-500",
      },
      orange: {
        bg: "bg-orange-50",
        text: "text-orange-600",
        border: "border-orange-200",
        progress: "bg-orange-500",
      },
      pink: {
        bg: "bg-pink-50",
        text: "text-pink-600",
        border: "border-pink-200",
        progress: "bg-pink-500",
      },
    };
    return colors[color] || colors.blue;
  };

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Brand Diagnostics
        </h1>
        <p className="text-gray-600">
          Comprehensive analysis of your brand health across multiple dimensions
        </p>
      </div>

      {/* Main Tabs */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-2">
        <div className="flex gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={`flex-1 px-6 py-3 rounded-lg font-semibold transition-all ${
                activeTab === tab.id
                  ? "bg-black text-black shadow-md"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Filter Dropdown */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center gap-4">
          <label className="text-sm font-semibold text-gray-700">
            Filter by {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}:
          </label>
          <select
            value={selectedFilter}
            onChange={(e) => setSelectedFilter(e.target.value)}
            className="flex-1 max-w-xs px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent"
          >
            {filterOptions[activeTab].map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
          <div className="ml-auto text-sm text-gray-500">
            Analyzing:{" "}
            <span className="font-semibold text-gray-900">
              {selectedFilter}
            </span>
          </div>
        </div>
      </div>

      {/* Overall Score Card */}
      <motion.div
        key={`${activeTab}-${selectedFilter}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-gray-900 to-gray-700 rounded-xl shadow-lg p-8 text-white"
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">Overall Health Score</h2>
            <p className="text-gray-300">
              For {selectedFilter} in{" "}
              {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} context
            </p>
          </div>
          <div className="text-center">
            <div className="text-6xl font-bold mb-2">80%</div>
            <div className="text-sm text-gray-300">Good Performance</div>
          </div>
        </div>
      </motion.div>

      {/* Diagnostic Sections Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {diagnosticSections.map((section, index) => {
          const Icon = section.icon;
          const colors = getColorClasses(section.color);

          return (
            <motion.div
              key={section.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
            >
              {/* Section Header */}
              <div className="flex items-center gap-3 mb-4">
                <div
                  className={`w-12 h-12 ${colors.bg} rounded-xl flex items-center justify-center`}
                >
                  <Icon className={`w-6 h-6 ${colors.text}`} />
                </div>
                <div className="flex-1">
                  <h3 className="font-bold text-gray-900">{section.title}</h3>
                  <p className="text-xs text-gray-500">{section.description}</p>
                </div>
              </div>

              {/* Score */}
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600">Score</span>
                  <span className={`text-2xl font-bold ${colors.text}`}>
                    {section.score}%
                  </span>
                </div>
                <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${section.score}%` }}
                    transition={{ duration: 1, delay: index * 0.1 + 0.3 }}
                    className={`h-full ${colors.progress} rounded-full`}
                  />
                </div>
              </div>

              {/* Quick Stats */}
              <div className="grid grid-cols-2 gap-3 pt-4 border-t border-gray-100">
                <div>
                  <div className="text-xs text-gray-500">vs Average</div>
                  <div className="text-sm font-semibold text-green-600">
                    +12%
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500">Trend</div>
                  <div className="text-sm font-semibold text-blue-600">
                    ↑ Growing
                  </div>
                </div>
              </div>

              {/* View Details Button */}
              <button className="w-full mt-4 px-4 py-2 bg-gray-50 hover:bg-gray-100 text-gray-700 text-sm font-medium rounded-lg transition-colors">
                View Details
              </button>
            </motion.div>
          );
        })}
      </div>

      {/* Insights Section */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">
          Key Insights for {selectedFilter}
        </h3>
        <div className="space-y-3">
          <div className="flex items-start gap-3 p-4 bg-green-50 rounded-lg border border-green-200">
            <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold text-gray-900">
                Strong Messaging Consistency
              </div>
              <div className="text-sm text-gray-600">
                Your brand maintains excellent consistency across channels in
                this segment
              </div>
            </div>
          </div>
          <div className="flex items-start gap-3 p-4 bg-orange-50 rounded-lg border border-orange-200">
            <AlertTriangle className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold text-gray-900">
                Opportunity in Specificity
              </div>
              <div className="text-sm text-gray-600">
                Consider adding more data-backed claims to improve credibility
                in this market
              </div>
            </div>
          </div>
          <div className="flex items-start gap-3 p-4 bg-red-50 rounded-lg border border-red-200">
            <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold text-gray-900">
                GAP Requires Attention
              </div>
              <div className="text-sm text-gray-600">
                Competitors are outperforming in innovation messaging for this
                segment
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
