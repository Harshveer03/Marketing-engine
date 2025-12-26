import React, { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  BarChart3,
  TrendingUp,
  AlertTriangle,
  AlertCircle,
  CheckCircle,
  Target,
  MessageSquare,
  Calendar,
  Activity,
  Download,
  X,
  ChevronRight,
  TrendingDown,
  Minus,
  Info,
  Filter,
  RefreshCw,
  Trophy,
  Wrench,
  Shield,
} from "lucide-react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
} from "recharts";

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
  const [activeView, setActiveView] = useState<"diagnostics" | "insights">("diagnostics");
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
    <div className="relative overflow-hidden">
      {/* Grid pattern background */}


      <div className="relative z-10 p-8 space-y-6">
        {/* View Toggle Button - At the very top */}
        <div className="flex justify-center mb-6">
          <div className="inline-flex bg-gray-100 rounded-full p-1">
            <button
              onClick={() => setActiveView("diagnostics")}
              style={
                activeView === "diagnostics"
                  ? { backgroundColor: "#000", color: "#fff" }
                  : {}
              }
              className={`px-8 py-2.5 rounded-full font-medium transition-all ${activeView === "diagnostics"
                ? "shadow-md"
                : "text-gray-700 hover:text-gray-900"
                }`}
            >
              Diagnostics
            </button>
            <button
              onClick={() => setActiveView("insights")}
              style={
                activeView === "insights"
                  ? { backgroundColor: "#000", color: "#fff" }
                  : {}
              }
              className={`px-8 py-2.5 rounded-full font-medium transition-all ${activeView === "insights"
                ? "shadow-md"
                : "text-gray-700 hover:text-gray-900"
                }`}
            >
              Insights
            </button>
          </div>
        </div>

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            {activeView === "diagnostics" ? "Brand Diagnostics" : "Brand Insights"}
          </h1>
          <p className="text-gray-600">
            {activeView === "diagnostics"
              ? "Comprehensive analysis of your brand health across multiple dimensions"
              : "Detailed analysis of your brand's performance across key diagnostic metrics"
            }
          </p>
        </div>

        {/* Diagnostics View */}
        {activeView === "diagnostics" && (
          <>
            {/* Performance Snapshot Section */}
            <div className="mb-2">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Performance Snapshot</h2>

              {/* Horizontal Scrollable Cards Container */}
              <div className="overflow-x-auto pb-4 -mx-2 px-2">
                <div
                  className="flex gap-4 lg:grid lg:gap-4 lg:overflow-visible"
                  style={{
                    gridTemplateColumns: 'repeat(7, minmax(0, 1fr))'
                  }}
                >
                  {diagnosticSections.map((section, index) => {
                    const Icon = section.icon;
                    const colors = getColorClasses(section.color);

                    // Determine trend direction based on score
                    const getTrendIcon = () => {
                      if (section.score >= 80) return <TrendingUp className="w-4 h-4 text-green-500" />;
                      if (section.score >= 65) return <Minus className="w-4 h-4 text-gray-400" />;
                      return <TrendingDown className="w-4 h-4 text-red-500" />;
                    };

                    const getTrendText = () => {
                      if (section.score >= 80) return "vs. Avg 75%";
                      if (section.score >= 65) return "vs. Avg 70%";
                      return "vs. Avg 75%";
                    };

                    const getTrendColor = () => {
                      if (section.score >= 80) return "text-green-600";
                      if (section.score >= 65) return "text-gray-600";
                      return "text-red-600";
                    };

                    return (
                      <motion.div
                        key={section.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className="bg-white shadow-sm border border-gray-200 p-4 hover:shadow-md transition-all flex-shrink-0 flex flex-col overflow-hidden"
                        style={{
                          width: '174px',
                          minWidth: '174px',
                          maxWidth: '174px',
                          height: '214px',
                          minHeight: '214px',
                          maxHeight: '214px'
                        }}
                      >
                        {/* Header with Icon and Trend - Fixed Height */}
                        <div className="flex items-start justify-between mb-3 h-8 flex-shrink-0">
                          <div className={`w-8 h-8 ${colors.bg} rounded-lg flex items-center justify-center flex-shrink-0`}>
                            <Icon className={`w-4 h-4 ${colors.text}`} />
                          </div>
                          <div className="flex-shrink-0">
                            {getTrendIcon()}
                          </div>
                        </div>

                        {/* Metric Name - Fixed Height - Centered */}
                        <h3 className="text-sm font-semibold text-gray-700 mb-1 h-10 line-clamp-2 flex-shrink-0 text-center">
                          {section.title}
                        </h3>

                        {/* Score - Fixed Height - Centered */}
                        <div className="mb-2 h-10 flex items-center justify-center flex-shrink-0">
                          <span className={`text-3xl font-bold ${colors.text} leading-none`}>
                            {section.score}%
                          </span>
                        </div>

                        {/* Comparison - Fixed Height - Centered */}
                        <div className={`text-xs font-medium mb-3 h-5 flex items-center justify-center flex-shrink-0 ${getTrendColor()}`}>
                          {getTrendText()}
                        </div>

                        {/* Description - Fixed Height - Centered */}
                        <p className="text-gray-500 line-clamp-2 h-8 flex-shrink-0 text-center" style={{ fontSize: '10px' }}>
                          {section.description}
                        </p>
                      </motion.div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Why this diagnostic matters Card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="mb-4 bg-gray-100 rounded-xl border border-gray-100 p-6"
            >
              <h3 className="text-base font-bold text-gray-900 mb-3">Why this diagnostic matters</h3>
              <p className="text-sm text-gray-700 mb-3">
                This canvas shows how clearly and consistently you show up in your market compared to direct competitors. Use it to:
              </p>
              <ul className="list-disc list-inside text-sm text-gray-700 space-y-1 ml-2">
                <li>Prioritize which gaps to close first.</li>
                <li>Align marketing, product, and sales around the same signals.</li>
                <li>Track the impact of launches and campaigns on market perception over time.</li>
              </ul>
            </motion.div>

            {/* Focus Lens Selector Section */}
            <div className="mb-8">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Focus Lens Selector</h2>

              {/* Category Tabs - First Row */}
              <div className="flex gap-3 flex-wrap mb-4">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => handleTabChange(tab.id)}
                    style={
                      activeTab === tab.id
                        ? { backgroundColor: "#000", color: "#fff" }
                        : {}
                    }
                    className={`px-6 py-2.5 rounded-full font-medium transition-all ${activeTab === tab.id
                      ? "shadow-md"
                      : "bg-white text-gray-700 border border-gray-300 hover:border-gray-400"
                      }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Filter Options - Second Row */}
              <div className="flex gap-3 flex-wrap">
                {filterOptions[activeTab].map((option) => (
                  <button
                    key={option}
                    onClick={() => setSelectedFilter(option)}
                    style={
                      selectedFilter === option
                        ? { backgroundColor: "#000", color: "#fff" }
                        : {}
                    }
                    className={`px-6 py-2.5 rounded-full font-medium transition-all whitespace-nowrap ${selectedFilter === option
                      ? "shadow-md"
                      : "bg-white text-gray-700 border border-gray-300 hover:border-gray-400"
                      }`}
                  >
                    {option}
                  </button>
                ))}
              </div>
            </div>

            {/* Three Cards Section - 2 Column Layout */}
            <div className="mb-12 pt-8">
              <div className="grid grid-cols-2 gap-4 w-full">
                {/* Left Column: Competitor Comparison Grid - 50% width */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="col-span-1 bg-white rounded-xl shadow-md border border-gray-200 p-6 h-full flex flex-col"
                >
                  <h3 className="text-lg font-bold mb-8 text-gray-900">Competitor Comparison Grid</h3>

                  {/* Comparison Table */}
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-gray-200">
                          <th className="text-left py-2 pr-2 font-semibold text-gray-600 text-xs">METRIC</th>
                          <th className="text-center py-2 px-2 font-semibold text-gray-900 text-xs">YOU</th>
                          <th className="text-center py-2 px-2 font-semibold text-gray-600 text-xs">COMPETITOR A</th>
                          <th className="text-center py-2 px-2 font-semibold text-gray-600 text-xs">COMPETITOR B</th>
                          <th className="text-center py-2 px-2 font-semibold text-gray-600 text-xs">COMPETITOR C</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b border-gray-100">
                          <td className="py-2 pr-2 text-gray-700 text-sm">Visibility</td>
                          <td className="text-center py-2 px-2 font-semibold text-gray-900">82%</td>
                          <td className="text-center py-2 px-2 text-gray-600">78%</td>
                          <td className="text-center py-2 px-2 text-gray-600">90%</td>
                          <td className="text-center py-2 px-2 text-gray-600">65%</td>
                        </tr>
                        <tr className="border-b border-gray-100">
                          <td className="py-2 pr-2 text-gray-700 text-sm">Authority</td>
                          <td className="text-center py-2 px-2 font-semibold text-gray-900">75%</td>
                          <td className="text-center py-2 px-2 text-gray-600">80%</td>
                          <td className="text-center py-2 px-2 text-gray-600">72%</td>
                          <td className="text-center py-2 px-2 text-gray-600">60%</td>
                        </tr>
                        <tr className="border-b border-gray-100">
                          <td className="py-2 pr-2 text-gray-700 text-sm">Content Coverage</td>
                          <td className="text-center py-2 px-2 font-semibold text-gray-900">88%</td>
                          <td className="text-center py-2 px-2 text-gray-600">85%</td>
                          <td className="text-center py-2 px-2 text-gray-600">75%</td>
                          <td className="text-center py-2 px-2 text-gray-600">55%</td>
                        </tr>
                        <tr className="border-b border-gray-100">
                          <td className="py-2 pr-2 text-gray-700 text-sm">Messaging Strength</td>
                          <td className="text-center py-2 px-2 font-semibold text-gray-900">70%</td>
                          <td className="text-center py-2 px-2 text-gray-600">72%</td>
                          <td className="text-center py-2 px-2 text-gray-600">68%</td>
                          <td className="text-center py-2 px-2 text-gray-600">50%</td>
                        </tr>
                        <tr className="border-b border-gray-100">
                          <td className="py-2 pr-2 text-gray-700 text-sm">Keyword Overlap</td>
                          <td className="text-center py-2 px-2 font-semibold text-gray-900">65%</td>
                          <td className="text-center py-2 px-2 text-gray-600">70%</td>
                          <td className="text-center py-2 px-2 text-gray-600">60%</td>
                          <td className="text-center py-2 px-2 text-gray-600">45%</td>
                        </tr>
                        <tr className="border-b border-gray-100">
                          <td className="py-2 pr-2 text-gray-700 text-sm">Engagement Footprint</td>
                          <td className="text-center py-2 px-2 font-semibold text-gray-900">78%</td>
                          <td className="text-center py-2 px-2 text-gray-600">75%</td>
                          <td className="text-center py-2 px-2 text-gray-600">85%</td>
                          <td className="text-center py-2 px-2 text-gray-600">62%</td>
                        </tr>
                        <tr className="border-b border-gray-100">
                          <td className="py-2 pr-2 text-gray-700 text-sm">Webwide Mentions</td>
                          <td className="text-center py-2 px-2 font-semibold text-gray-900">92%</td>
                          <td className="text-center py-2 px-2 text-gray-600">89%</td>
                          <td className="text-center py-2 px-2 text-gray-600">80%</td>
                          <td className="text-center py-2 px-2 text-gray-600">70%</td>
                        </tr>
                        <tr>
                          <td className="py-2 pr-2 text-gray-700 text-sm">Trend Position</td>
                          <td className="text-center py-2 px-2 font-semibold text-green-600">Leader</td>
                          <td className="text-center py-2 px-2 text-gray-600">Follower</td>
                          <td className="text-center py-2 px-2 text-gray-600">Lagging</td>
                          <td className="text-center py-2 px-2 text-gray-600">Disruptor</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  {/* Visual Insights - Mini Bar Charts */}
                  <div className="mt-12 pt-8 border-t border-gray-200">
                    <h4 className="text-sm font-bold mb-8 text-gray-700">Quick Comparison</h4>
                    <div className="space-y-2">
                      {[
                        { metric: 'Visibility', you: 82, best: 90, competitor: 'Comp B' },
                        { metric: 'Authority', you: 75, best: 80, competitor: 'Comp A' },
                        { metric: 'Content', you: 88, best: 88, competitor: 'You' },
                        { metric: 'Engagement', you: 78, best: 85, competitor: 'Comp B' },
                      ].map((item, idx) => (
                        <div key={idx} className="flex items-center gap-2">
                          <div className="w-20 text-xs text-gray-600 truncate">{item.metric}</div>
                          <div className="flex-1 flex items-center gap-1">
                            {/* You bar */}
                            <div className="relative flex-1 h-5 bg-gray-100 rounded overflow-hidden">
                              <div
                                className="h-full transition-all"
                                style={{ width: `${item.you}%`, backgroundColor: '#2563EB' }}
                              />
                              <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white">
                                {item.you}%
                              </span>
                            </div>
                            {/* Best competitor bar */}
                            <div className="relative flex-1 h-5 bg-gray-100 rounded overflow-hidden">
                              <div
                                className="h-full transition-all"
                                style={{ width: `${item.best}%`, backgroundColor: '#4B5563' }}
                              />
                              <span className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
                                {item.best}%
                              </span>
                            </div>
                          </div>
                          {/* Indicator */}
                          {item.you >= item.best ? (
                            <div className="w-4 h-4 rounded-full bg-green-500 flex items-center justify-center">
                              <span className="text-green-600 text-xs">✓</span>
                            </div>
                          ) : (
                            <div className="w-4 h-4 rounded-full bg-orange-500 flex items-center justify-center">
                              <span className="text-orange-600 text-xs">!</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                    <div className="mt-2 flex items-center gap-3 text-xs text-gray-500">
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-3 bg-blue-600 rounded"></div>
                        <span>You</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-3 bg-gray-600 rounded"></div>
                        <span>Best Competitor</span>
                      </div>
                    </div>
                  </div>

                  {/* Summary Section */}
                  <div className="mt-12 pt-8 border-t border-gray-200">
                    <h4 className="text-sm font-bold mb-8 text-gray-700">Summary</h4>
                    <ul className="space-y-1.5 text-xs text-gray-600">
                      <li>• Strong in brand mentions and content, leading in many areas.</li>
                      <li>• Good visibility, slightly higher authority in key topics.</li>
                      <li>• High visibility but weaker content coverage, focused on niche.</li>
                      <li>• Emerging brand, significant growth potential in specific areas.</li>
                    </ul>
                  </div>
                </motion.div>

                {/* Right Column: Two Cards Stacked - 50% width */}
                <div className="col-span-1 space-y-4">
                  {/* Card 2: Trend Position Mini Panel */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="bg-white rounded-xl shadow-md border border-gray-200 p-6"
                  >
                    {/* Header Section */}
                    <div className="mb-6">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <h3 className="text-xl font-bold text-gray-900 mb-0.5">Trend position panel</h3>
                          <p className="text-sm text-gray-600">
                            Where each brand sits on category trend dynamics.
                          </p>
                        </div>
                        <span className="bg-gray-100 text-gray-600 text-xs font-medium px-3 py-1 rounded-full whitespace-nowrap ml-4">
                          DiagnosticResult.trendPanel
                        </span>
                      </div>
                    </div>

                    {/* 2x2 Quadrant Grid */}
                    <div className="grid grid-cols-2 gap-4">
                      {/* Leader - Top Left */}
                      <motion.div
                        whileHover={{ scale: 1.02 }}
                        className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm hover:shadow-md transition-all h-[200px] flex flex-col"
                      >
                        <h4 className="text-base font-bold text-gray-900 mb-3">Leader</h4>
                        <p className="text-sm text-gray-600 mb-5 flex-shrink-0 leading-relaxed">
                          Sets the narrative and is referenced as a category-defining voice.
                        </p>
                        <div className="mt-auto">
                          <div className="flex items-center justify-between gap-4">
                            <div className="flex items-center gap-2">
                              <span className="text-sm text-gray-500">•</span>
                              <span className="text-sm text-gray-700">Comp A</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 bg-blue-600 rounded-full flex-shrink-0"></div>
                              <span className="text-sm bg-gray-900 text-white px-2.5 py-1 rounded-md font-medium">You</span>
                            </div>
                          </div>
                        </div>
                      </motion.div>

                      {/* Disruptor - Top Right */}
                      <motion.div
                        whileHover={{ scale: 1.02 }}
                        className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm hover:shadow-md transition-all h-[200px] flex flex-col"
                      >
                        <h4 className="text-base font-bold text-gray-900 mb-3">Disruptor</h4>
                        <p className="text-sm text-gray-600 mb-5 flex-shrink-0 leading-relaxed">
                          Introduces new angles and contrarian takes that reshape demand.
                        </p>
                        <div className="mt-auto space-y-2.5">
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-gray-500">•</span>
                            <span className="text-sm text-gray-700">Comp B</span>
                          </div>
                        </div>
                      </motion.div>

                      {/* Follower - Bottom Left */}
                      <motion.div
                        whileHover={{ scale: 1.02 }}
                        className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm hover:shadow-md transition-all h-[200px] flex flex-col"
                      >
                        <h4 className="text-base font-bold text-gray-900 mb-3">Follower</h4>
                        <p className="text-sm text-gray-600 mb-5 flex-shrink-0 leading-relaxed">
                          Joins trends late and mainly echoes existing narratives.
                        </p>
                        <div className="mt-auto space-y-2.5">
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-gray-500">•</span>
                            <span className="text-sm text-gray-700">Comp C</span>
                          </div>
                        </div>
                      </motion.div>

                      {/* Lagging - Bottom Right */}
                      <motion.div
                        whileHover={{ scale: 1.02 }}
                        className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm hover:shadow-md transition-all h-[200px] flex flex-col"
                      >
                        <h4 className="text-base font-bold text-gray-900 mb-3">Lagging</h4>
                        <p className="text-sm text-gray-600 mb-5 flex-shrink-0 leading-relaxed">
                          Rarely shows up in category conversations or emerging themes.
                        </p>
                        <div className="mt-auto space-y-2.5">
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-gray-500">•</span>
                            <span className="text-sm text-gray-700">Long tail</span>
                          </div>
                        </div>
                      </motion.div>
                    </div>
                  </motion.div>

                  {/* Card 3: Diagnostic Radar */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="bg-white rounded-xl shadow-md border border-gray-200 p-6"
                  >
                    <h3 className="text-xl font-bold mb-6 text-gray-900">Diagnostic Radar</h3>

                    {/* Radar Chart */}
                    <ResponsiveContainer width="100%" height={500}>
                      <RadarChart data={[
                        { metric: 'Clarity', you: 85, competitorA: 75, competitorB: 80, competitorC: 65 },
                        { metric: 'Specificity', you: 72, competitorA: 78, competitorB: 70, competitorC: 60 },
                        { metric: 'Relevance', you: 88, competitorA: 82, competitorB: 85, competitorC: 70 },
                        { metric: 'GAP', you: 65, competitorA: 70, competitorB: 60, competitorC: 55 },
                        { metric: 'Messaging', you: 90, competitorA: 85, competitorB: 88, competitorC: 75 },
                        { metric: 'Posting', you: 78, competitorA: 80, competitorB: 75, competitorC: 70 },
                        { metric: 'Trend', you: 82, competitorA: 75, competitorB: 78, competitorC: 68 },
                      ]}>
                        <PolarGrid stroke="#E5E7EB" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#6B7280', fontSize: 12 }} />
                        <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#9CA3AF', fontSize: 10 }} />
                        <Radar name="You" dataKey="you" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.5} />
                        <Radar name="Competitor A" dataKey="competitorA" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.3} />
                        <Radar name="Competitor B" dataKey="competitorB" stroke="#10B981" fill="#10B981" fillOpacity={0.3} />
                        <Radar name="Competitor C" dataKey="competitorC" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.3} />
                        <Legend
                          wrapperStyle={{ paddingTop: '20px' }}
                          iconType="circle"
                          formatter={(value) => <span style={{ color: '#374151', fontSize: '12px' }}>{value}</span>}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </motion.div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Insights View */}
        {activeView === "insights" && (
          <>
            <h3 className="text-lg font-bold text-gray-900 mb-4">Gap Card Panel</h3>

            {/* Gap Cards Panel - Brand Insights */}
            <div className="grid grid-cols-4 gap-4 mb-10">
              {/* Card 1: Clarity */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0 }}
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow flex flex-col"
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-base font-bold text-gray-900">Clarity</h3>
                  <span className="px-2.5 py-1 bg-yellow-100 text-yellow-700 text-xs font-semibold rounded-full">Low</span>
                </div>

                {/* Score Section */}
                <div className="mb-4">
                  <div className="text-3xl font-bold text-blue-600 mb-1">78%</div>
                  <div className="text-sm text-gray-600 mb-0.5">vs. Avg 65%</div>
                  <div className="text-sm text-gray-600">Confidence: 90%</div>
                </div>

                {/* Description */}
                <p className="text-sm text-gray-700 mb-5 leading-relaxed">
                  Our brand's messaging is generally clear and understandable to our target audience. However, specific product feature explanations could benefit from more concise language and visual aids to enhance immediate comprehension.
                </p>

                {/* Webwide Highlight */}
                <div className="mb-4 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Webwide Highlight</h4>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    Customer praise our "easy-to-understand" onboarding process, but some support tickets indicate confusion on advanced features.
                  </p>
                </div>

                {/* Competitor Examples */}
                <div className="mb-5 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Competitor Examples</h4>
                  <ul className="text-sm text-gray-600 space-y-1.5 list-disc list-inside">
                    <li>Competitor A uses animated explainers for complex topics</li>
                    <li>Competitor B provides interactive tutorials for every new feature</li>
                  </ul>
                </div>

                {/* Mentions & Trend */}
                <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200 text-sm">
                  <span className="text-gray-600">Mentions: <span className="font-semibold text-gray-900">15,400</span></span>
                  <span className="text-gray-600">Trend: <span className="font-semibold text-gray-900">Stable</span></span>
                </div>

                {/* Platform Distribution */}
                <div className="mb-5">
                  <h4 className="text-xs font-bold text-gray-900 mb-3 uppercase tracking-wide">Platform Distribution</h4>
                  <div className="flex flex-wrap gap-2">
                    <span className="text-sm bg-blue-50 text-blue-700 px-3 py-1.5 rounded-md font-medium">LinkedIn (40%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Blog (30%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Twitter (20%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Forums (10%)</span>
                  </div>
                </div>

                {/* Action Button */}
                <button className="w-full mt-auto px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors">
                  View deep scan
                </button>
              </motion.div>

              {/* Card 2: Specificity */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow flex flex-col"
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-base font-bold text-gray-900">Specificity</h3>
                  <span className="px-2.5 py-1 bg-orange-100 text-orange-700 text-xs font-semibold rounded-full">Medium</span>
                </div>

                <div className="mb-4">
                  <div className="text-3xl font-bold text-purple-600 mb-1">72%</div>
                  <div className="text-sm text-gray-600 mb-0.5">vs. Avg 75%</div>
                  <div className="text-sm text-gray-600">Confidence: 85%</div>
                </div>

                <p className="text-sm text-gray-700 mb-5 leading-relaxed">
                  Our content, while broad, sometimes lacks the depth and specific details that top competitors provide. This can make it harder for technical users to find precise answers, potentially leading to increased bounce rates.
                </p>

                <div className="mb-4 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Webwide Highlight</h4>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    Several industry reviews mention our content as "informative but high-level" contrasting with competitors' "data-rich analyses."
                  </p>
                </div>

                <div className="mb-5 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Competitor Examples</h4>
                  <ul className="text-sm text-gray-600 space-y-1.5 list-disc list-inside">
                    <li>Competitor A publishes detailed case studies with real figures</li>
                    <li>Competitor B offers comprehensive technical whitepapers</li>
                  </ul>
                </div>

                <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200 text-sm">
                  <span className="text-gray-600">Mentions: <span className="font-semibold text-gray-900">12,100</span></span>
                  <span className="text-gray-600">Trend: <span className="font-semibold text-red-600">Declining</span></span>
                </div>

                <div className="mb-5">
                  <h4 className="text-xs font-bold text-gray-900 mb-3 uppercase tracking-wide">Platform Distribution</h4>
                  <div className="flex flex-wrap gap-2">
                    <span className="text-sm bg-blue-50 text-blue-700 px-3 py-1.5 rounded-md font-medium">Blog (50%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">LinkedIn (25%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Research Sites (15%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">News (10%)</span>
                  </div>
                </div>

                <button className="w-full mt-auto px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors">
                  View deep scan
                </button>
              </motion.div>

              {/* Card 3: Relevance */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow flex flex-col"
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-base font-bold text-gray-900">Relevance</h3>
                  <span className="px-2.5 py-1 bg-yellow-100 text-yellow-700 text-xs font-semibold rounded-full">Low</span>
                </div>

                <div className="mb-4">
                  <div className="text-3xl font-bold text-green-600 mb-1">88%</div>
                  <div className="text-sm text-gray-600 mb-0.5">vs. Avg 80%</div>
                  <div className="text-sm text-gray-600">Confidence: 92%</div>
                </div>

                <p className="text-sm text-gray-700 mb-5 leading-relaxed">
                  Our content strategy aligns closely with current market interests. We consistently address pain points and offer solutions that resonate, as evidenced by high engagement rates on core topics.
                </p>

                <div className="mb-4 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Webwide Highlight</h4>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    Sentiment analysis indicates strong positive correlation between our content and audience needs, often cited as "spot-on."
                  </p>
                </div>

                <div className="mb-5 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Competitor Examples</h4>
                  <ul className="text-sm text-gray-600 space-y-1.5 list-disc list-inside">
                    <li>Competitor A recently launched a series on niche industry challenges</li>
                    <li>Competitor B partners with influencers for trending topics</li>
                  </ul>
                </div>

                <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200 text-sm">
                  <span className="text-gray-600">Mentions: <span className="font-semibold text-gray-900">18,900</span></span>
                  <span className="text-gray-600">Trend: <span className="font-semibold text-green-600">Growing</span></span>
                </div>

                <div className="mb-5">
                  <h4 className="text-xs font-bold text-gray-900 mb-3 uppercase tracking-wide">Platform Distribution</h4>
                  <div className="flex flex-wrap gap-2">
                    <span className="text-sm bg-blue-50 text-blue-700 px-3 py-1.5 rounded-md font-medium">LinkedIn (25%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Twitter (30%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Forums (20%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">News (15%)</span>
                  </div>
                </div>

                <button className="w-full mt-auto px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors">
                  View deep scan
                </button>
              </motion.div>

              {/* Card 4: Influence */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow flex flex-col"
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-base font-bold text-gray-900">Influence</h3>
                  <span className="px-2.5 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full">High</span>
                </div>

                <div className="mb-4">
                  <div className="text-3xl font-bold text-indigo-600 mb-1">68%</div>
                  <div className="text-sm text-gray-600 mb-0.5">vs. Avg 70%</div>
                  <div className="text-sm text-gray-600">Confidence: 75%</div>
                </div>

                <p className="text-sm text-gray-700 mb-5 leading-relaxed">
                  While we have a solid audience, our brand's ability to drive mass conversation and shape opinions is moderate. We need to boost thought leadership through strategic partnerships and expert content contributors.
                </p>

                <div className="mb-4 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Webwide Highlight</h4>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    Mentions in tier-1 media are often reactive rather than proactive, appearing in discussions initiated by others.
                  </p>
                </div>

                <div className="mb-5 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Competitor Examples</h4>
                  <ul className="text-sm text-gray-600 space-y-1.5 list-disc list-inside">
                    <li>Competitor A regularly hosts industry webinars and expert panels</li>
                    <li>Competitor B's executives are frequently quoted in major publications</li>
                  </ul>
                </div>

                <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200 text-sm">
                  <span className="text-gray-600">Mentions: <span className="font-semibold text-gray-900">9,500</span></span>
                  <span className="text-gray-600">Trend: <span className="font-semibold text-gray-900">Stagnant</span></span>
                </div>

                <div className="mb-5">
                  <h4 className="text-xs font-bold text-gray-900 mb-3 uppercase tracking-wide">Platform Distribution</h4>
                  <div className="flex flex-wrap gap-2">
                    <span className="text-sm bg-blue-50 text-blue-700 px-3 py-1.5 rounded-md font-medium">Blog (50%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">News (20%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Podcasts (15%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Twitter (15%)</span>
                  </div>
                </div>

                <button className="w-full mt-auto px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors">
                  View deep scan
                </button>
              </motion.div>

              {/* Card 5: Impact */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow flex flex-col"
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-base font-bold text-gray-900">Impact</h3>
                  <span className="px-2.5 py-1 bg-red-100 text-red-700 text-xs font-semibold rounded-full">Critical</span>
                </div>

                <div className="mb-4">
                  <div className="text-3xl font-bold text-pink-600 mb-1">70%</div>
                  <div className="text-sm text-gray-600 mb-0.5">vs. Avg 82%</div>
                  <div className="text-sm text-gray-600">Confidence: 70%</div>
                </div>

                <p className="text-sm text-gray-700 mb-5 leading-relaxed">
                  Our content struggles to translate into measurable business outcomes. Customer testimonials are underutilized and clear calls-to-action are missing, making it compelling enough. This directly affects lead generation and sales.
                </p>

                <div className="mb-4 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Webwide Highlight</h4>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    Analytics show high traffic to content, but low conversion rates compared to industry benchmarks. Missing strong social proof.
                  </p>
                </div>

                <div className="mb-5 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Competitor Examples</h4>
                  <ul className="text-sm text-gray-600 space-y-1.5 list-disc list-inside">
                    <li>Competitor A prominently features client success stories and testimonials</li>
                    <li>Competitor B uses interactive ROI calculators on product pages</li>
                  </ul>
                </div>

                <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200 text-sm">
                  <span className="text-gray-600">Mentions: <span className="font-semibold text-gray-900">11,000</span></span>
                  <span className="text-gray-600">Trend: <span className="font-semibold text-red-600">Declining</span></span>
                </div>

                <div className="mb-5">
                  <h4 className="text-xs font-bold text-gray-900 mb-3 uppercase tracking-wide">Platform Distribution</h4>
                  <div className="flex flex-wrap gap-2">
                    <span className="text-sm bg-blue-50 text-blue-700 px-3 py-1.5 rounded-md font-medium">Website (60%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Email (20%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">LinkedIn (10%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Ads (10%)</span>
                  </div>
                </div>

                <button className="w-full mt-auto px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors">
                  View deep scan
                </button>
              </motion.div>

              {/* Card 6: Messaging consistency */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow flex flex-col"
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-base font-bold text-gray-900">Messaging consistency</h3>
                  <span className="px-2.5 py-1 bg-yellow-100 text-yellow-700 text-xs font-semibold rounded-full">Low</span>
                </div>

                <div className="mb-4">
                  <div className="text-3xl font-bold text-teal-600 mb-1">94%</div>
                  <div className="text-sm text-gray-600 mb-0.5">vs. Avg 88%</div>
                  <div className="text-sm text-gray-600">Confidence: 95%</div>
                </div>

                <p className="text-sm text-gray-700 mb-5 leading-relaxed">
                  Our brand maintains a highly consistent voice and messaging across all communication channels. This fosters strong brand recognition and trust among our audience, reinforcing our core values effectively.
                </p>

                <div className="mb-4 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Webwide Highlight</h4>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    Feedback consistently praises our "unwavering brand identity" and clear value proposition.
                  </p>
                </div>

                <div className="mb-5 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Competitor Examples</h4>
                  <ul className="text-sm text-gray-600 space-y-1.5 list-disc list-inside">
                    <li>Competitor A shows slight variations in tone across social media teams</li>
                    <li>Competitor B recently rebranded, leading to temporary inconsistencies</li>
                  </ul>
                </div>

                <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200 text-sm">
                  <span className="text-gray-600">Mentions: <span className="font-semibold text-gray-900">16,200</span></span>
                  <span className="text-gray-600">Trend: <span className="font-semibold text-gray-900">Stable</span></span>
                </div>

                <div className="mb-5">
                  <h4 className="text-xs font-bold text-gray-900 mb-3 uppercase tracking-wide">Platform Distribution</h4>
                  <div className="flex flex-wrap gap-2">
                    <span className="text-sm bg-blue-50 text-blue-700 px-3 py-1.5 rounded-md font-medium">All Channels (100%)</span>
                  </div>
                </div>

                <button className="w-full mt-auto px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors">
                  View deep scan
                </button>
              </motion.div>

              {/* Card 7: Posting consistency */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow flex flex-col"
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-base font-bold text-gray-900">Posting consistency</h3>
                  <span className="px-2.5 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full">High</span>
                </div>

                <div className="mb-4">
                  <div className="text-3xl font-bold text-orange-600 mb-1">60%</div>
                  <div className="text-sm text-gray-600 mb-0.5">vs. Avg 70%</div>
                  <div className="text-sm text-gray-600">Confidence: 88%</div>
                </div>

                <p className="text-sm text-gray-700 mb-5 leading-relaxed">
                  Our publishing schedule is irregular across key platforms, leading to missed opportunities for audience engagement and reduced organic reach. A more predictable and frequent content cadence is needed.
                </p>

                <div className="mb-4 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Webwide Highlight</h4>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    Audience polls express a desire for more frequent updates and content releases.
                  </p>
                </div>

                <div className="mb-5 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Competitor Examples</h4>
                  <ul className="text-sm text-gray-600 space-y-1.5 list-disc list-inside">
                    <li>Competitor A posts daily on LinkedIn and weekly on their blog</li>
                    <li>Competitor B has a strict content calendar shared with their audience</li>
                  </ul>
                </div>

                <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200 text-sm">
                  <span className="text-gray-600">Mentions: <span className="font-semibold text-gray-900">8,800</span></span>
                  <span className="text-gray-600">Trend: <span className="font-semibold text-gray-900">Stagnant</span></span>
                </div>

                <div className="mb-5">
                  <h4 className="text-xs font-bold text-gray-900 mb-3 uppercase tracking-wide">Platform Distribution</h4>
                  <div className="flex flex-wrap gap-2">
                    <span className="text-sm bg-blue-50 text-blue-700 px-3 py-1.5 rounded-md font-medium">Social Media (50%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Blog (30%)</span>
                    <span className="text-sm bg-gray-100 text-gray-700 px-3 py-1.5 rounded-md font-medium">Email (20%)</span>
                  </div>
                </div>

                <button className="w-full mt-auto px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors">
                  View deep scan
                </button>
              </motion.div>

              {/* Card 8: Trend position */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow flex flex-col"
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-base font-bold text-gray-900">Trend position</h3>
                  <span className="px-2.5 py-1 bg-yellow-100 text-yellow-700 text-xs font-semibold rounded-full">Low</span>
                </div>

                <div className="mb-4">
                  <div className="text-3xl font-bold text-cyan-600 mb-1">75%</div>
                  <div className="text-sm text-gray-600 mb-0.5">vs. Avg 68%</div>
                  <div className="text-sm text-gray-600">Confidence: 85%</div>
                </div>

                <p className="text-sm text-gray-700 mb-5 leading-relaxed">
                  We are generally well-positioned in emerging industry trends, often being early adopters and contributors. This proactive stance helps establish our reputation as an innovative leader in the market.
                </p>

                <div className="mb-4 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Webwide Highlight</h4>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    Our brand is frequently cited in discussions about future industry directions and technological advancements.
                  </p>
                </div>

                <div className="mb-5 bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">Competitor Examples</h4>
                  <ul className="text-sm text-gray-600 space-y-1.5 list-disc list-inside">
                    <li>Competitor A is slower to adopt new trends, focusing on established markets</li>
                    <li>Competitor B is attempting to pivot into new trends with mixed success</li>
                  </ul>
                </div>

                <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200 text-sm">
                  <span className="text-gray-600">Mentions: <span className="font-semibold text-gray-900">14,000</span></span>
                  <span className="text-gray-600">Trend: <span className="font-semibold text-green-600">Growing</span></span>
                </div>

                <div className="mb-5">
                  <h4 className="text-xs font-bold text-gray-900 mb-3 uppercase tracking-wide">Platform Distribution</h4>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="text-sm">
                      <span className="font-semibold text-gray-900">Tech Blogs</span>
                      <br />
                      <span className="text-gray-600">(30%)</span>
                    </div>
                    <div className="text-sm">
                      <span className="font-semibold text-gray-900">Conferences</span>
                      <br />
                      <span className="text-gray-600">(30%)</span>
                    </div>
                    <div className="text-sm">
                      <span className="font-semibold text-gray-900">Industry Reports</span>
                      <br />
                      <span className="text-gray-600">(20%)</span>
                    </div>
                    <div className="text-sm">
                      <span className="font-semibold text-gray-900">Forums</span>
                      <br />
                      <span className="text-gray-600">(10%)</span>
                    </div>
                  </div>
                </div>

                <button className="w-full mt-auto px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors">
                  View deep scan
                </button>
              </motion.div>
            </div>

            <h3 className="text-lg font-bold text-gray-900 mb-4">Gap Priority Map</h3>

            {/* Gap Priority Map - 2x2 Quadrant Matrix */}
            <div className="mb-8 bg-white rounded-xl shadow-sm border border-gray-200 p-8">
              <div className="relative" style={{ height: '640px', paddingLeft: '50px', paddingBottom: '50px' }}>
                {/* Y-axis Label (Impact) - Outside on the left */}
                <div className="absolute" style={{ left: '0px', top: '50%', transform: 'translateY(-50%) rotate(-90deg)' }}>
                  <span className="text-base font-semibold text-gray-700">Impact</span>
                </div>

                {/* Quadrant Grid */}
                <div className="absolute grid grid-cols-2 grid-rows-2 border border-gray-300" style={{ left: '50px', top: '0', right: '0', bottom: '50px' }}>
                  {/* Top-left Quadrant */}
                  <div className="border-r border-b border-gray-300 p-6 relative">
                    <span className="text-sm text-gray-600 font-medium absolute top-4 left-4">High Impact, Low Urgency</span>
                    {/* Data Point: Engage with Industry Influencers */}
                    <div className="absolute" style={{ top: '50%', left: '30%' }}>
                      <div className="relative group">
                        <div className="w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-lg"></div>
                        <div className="absolute left-6 -top-2 whitespace-nowrap text-sm text-gray-800 font-medium">
                          Engage with Industry Influencers
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Top-right Quadrant */}
                  <div className="border-b border-gray-300 p-6 relative">
                    <span className="text-sm text-gray-600 font-medium absolute top-4 right-4">High Impact, High Urgency</span>
                    {/* Data Point: Update Privacy Policy */}
                    <div className="absolute" style={{ top: '25%', right: '20%' }}>
                      <div className="relative group">
                        <div className="w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-lg"></div>
                        <div className="absolute right-6 -top-2 whitespace-nowrap text-sm text-gray-800 font-medium text-right">
                          Update Privacy Policy
                        </div>
                      </div>
                    </div>
                    {/* Data Point: Audit Website Accessibility */}
                    <div className="absolute" style={{ top: '55%', right: '25%' }}>
                      <div className="relative group">
                        <div className="w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-lg"></div>
                        <div className="absolute right-6 -top-2 whitespace-nowrap text-sm text-gray-800 font-medium text-right">
                          Audit Website Accessibility
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Bottom-left Quadrant */}
                  <div className="border-r border-gray-300 p-6 relative">
                    <span className="text-sm text-gray-600 font-medium absolute bottom-4 left-4">Low Impact, Low Urgency</span>
                  </div>

                  {/* Bottom-right Quadrant */}
                  <div className="p-6 relative">
                    <span className="text-sm text-gray-600 font-medium absolute bottom-4 right-4">Low Impact, High Urgency</span>
                    {/* Data Point: Launch Customer Testimonial Campaign */}
                    <div className="absolute" style={{ bottom: '60%', right: '40%' }}>
                      <div className="relative group">
                        <div className="w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-lg"></div>
                        <div className="absolute left-6 -top-2 whitespace-nowrap text-sm text-gray-800 font-medium">
                          Launch Customer Testimonial Campaign
                        </div>
                      </div>
                    </div>
                    {/* Data Point: Standardize Social Media Bios */}
                    <div className="absolute" style={{ bottom: '35%', right: '25%' }}>
                      <div className="relative group">
                        <div className="w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-lg"></div>
                        <div className="absolute left-6 -top-2 whitespace-nowrap text-sm text-gray-800 font-medium">
                          Standardize Social Media Bios
                        </div>
                      </div>
                    </div>
                    {/* Data Point: Optimize Blog SEO */}
                    <div className="absolute" style={{ bottom: '15%', right: '35%' }}>
                      <div className="relative group">
                        <div className="w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-lg"></div>
                        <div className="absolute left-6 -top-2 whitespace-nowrap text-sm text-gray-800 font-medium">
                          Optimize Blog SEO
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* X-axis Label (Urgency) - Outside at the bottom center */}
                <div className="absolute" style={{ bottom: '0px', left: '50%', transform: 'translateX(-50%)' }}>
                  <span className="text-base font-semibold text-gray-700">Urgency</span>
                </div>
              </div>
            </div>

            <h3 className="text-lg font-bold text-gray-900 mb-4">Strategic Paths</h3>

            {/* Strategic Paths - SWOT Analysis Grid */}
            <div className="grid grid-cols-4 gap-4 mb-10">
              {/* Strengths Column */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow duration-200 flex flex-col h-full">
                <div className="flex items-center gap-3 mb-4 pb-4 border-b border-gray-100">
                  <div className="p-2 bg-green-50 rounded-lg">
                    <Trophy className="w-5 h-5 text-green-600" />
                  </div>
                  <h4 className="text-lg font-bold text-gray-900">Strengths</h4>
                </div>
                <div className="space-y-6 flex-1">
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Amplify positive customer testimonials across all platforms.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Leverage existing satisfaction. Boost social proof.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Ongoing
                    </span>
                  </div>
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Publish in-depth case studies based on successful client projects.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Showcase expertise and impact.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Q3
                    </span>
                  </div>
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Host expert webinars leveraging internal thought leaders.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Reinforce knowledge authority.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Monthly
                    </span>
                  </div>
                </div>
              </div>

              {/* Weaknesses Column */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow duration-200 flex flex-col h-full">
                <div className="flex items-center gap-3 mb-4 pb-4 border-b border-gray-100">
                  <div className="p-2 bg-orange-50 rounded-lg">
                    <Wrench className="w-5 h-5 text-orange-600" />
                  </div>
                  <h4 className="text-lg font-bold text-gray-900">Weaknesses</h4>
                </div>
                <div className="space-y-6 flex-1">
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Conduct a comprehensive SEO audit for technical issues.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Identify and fix underlying ranking barriers.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Next 4 weeks
                    </span>
                  </div>
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Develop a focused content calendar on missing high-value topics.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Fill content gaps and improve relevance.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Ongoing
                    </span>
                  </div>
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Invest in PR to secure media mentions and backlinks.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Boost domain authority and credibility.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Next 6 months
                    </span>
                  </div>
                </div>
              </div>

              {/* Opportunities Column */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow duration-200 flex flex-col h-full">
                <div className="flex items-center gap-3 mb-4 pb-4 border-b border-gray-100">
                  <div className="p-2 bg-blue-50 rounded-lg">
                    <Target className="w-5 h-5 text-blue-600" />
                  </div>
                  <h4 className="text-lg font-bold text-gray-900">Opportunities</h4>
                </div>
                <div className="space-y-6 flex-1">
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Explore partnerships with complementary businesses.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Expand reach to new audiences.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Q4
                    </span>
                  </div>
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Develop content for emerging platforms like TikTok or Threads.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Capture younger demographics.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Next 3 months
                    </span>
                  </div>
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Target international markets with localized content.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Global expansion potential.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Q4
                    </span>
                  </div>
                </div>
              </div>

              {/* Threats Column */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow duration-200 flex flex-col h-full">
                <div className="flex items-center gap-3 mb-4 pb-4 border-b border-gray-100">
                  <div className="p-2 bg-red-50 rounded-lg">
                    <Shield className="w-5 h-5 text-red-600" />
                  </div>
                  <h4 className="text-lg font-bold text-gray-900">Threats</h4>
                </div>
                <div className="space-y-6 flex-1">
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Monitor competitor ad spend and adjust strategy.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Counter aggressive campaigns.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Ongoing
                    </span>
                  </div>
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Diversify traffic sources beyond main channels.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Mitigate risk from algorithm changes.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Q3
                    </span>
                  </div>
                  <div>
                    <p className="text-sm font-bold text-gray-900 leading-snug mb-1.5">
                      Strengthen cybersecurity measures and data privacy protocols.
                    </p>
                    <p className="text-sm text-gray-600 leading-relaxed mb-2">
                      Protect against breaches and regulatory fines.
                    </p>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Timeline: Next 3 months
                    </span>
                  </div>
                </div>
              </div>
            </div>


            {/* Unified Action Priority List */}
            <div className="mt-12">
              <h2 className="text-xl font-bold text-gray-900 mb-6">Unified Action Priority List</h2>

              <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                <table className="w-full">
                  <thead className="bg-gray-50 border-b border-gray-200">
                    <tr>
                      <th className="text-center px-6 py-3 text-xs font-semibold text-gray-600 uppercase tracking-wider">Action</th>
                      <th className="text-center px-4 py-3 text-xs font-semibold text-gray-600 uppercase tracking-wider">Urg.</th>
                      <th className="text-center px-4 py-3 text-xs font-semibold text-gray-600 uppercase tracking-wider">Eff.</th>
                      <th className="text-center px-4 py-3 text-xs font-semibold text-gray-600 uppercase tracking-wider">Conf.</th>
                      <th className="text-center px-4 py-3 text-xs font-semibold text-gray-600 uppercase tracking-wider">Strat. Edge</th>
                      <th className="text-center px-6 py-3 text-xs font-semibold text-gray-600 uppercase tracking-wider">GAP Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {/* Row 1: Update Privacy Policy */}
                    <tr className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 text-center">
                        <div>
                          <a href="#" className="text-sm font-medium text-blue-600 hover:text-blue-700">Update Privacy Policy</a>
                          <div className="mt-1">
                            <span className="inline-block px-2 py-0.5 bg-red-100 text-red-700 text-xs font-semibold rounded">Immediate</span>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">5</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">2</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">5</td>
                      <td className="px-4 py-4 text-center">
                        <CheckCircle className="w-5 h-5 text-green-500 inline-block" />
                      </td>
                      <td className="px-6 py-4 text-center text-sm font-semibold text-gray-900">9.2</td>
                    </tr>

                    {/* Row 2: Audit Website Accessibility */}
                    <tr className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 text-center">
                        <div>
                          <a href="#" className="text-sm font-medium text-blue-600 hover:text-blue-700">Audit Website Accessibility</a>
                          <div className="mt-1">
                            <span className="inline-block px-2 py-0.5 bg-orange-100 text-orange-700 text-xs font-semibold rounded">High</span>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">4</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">4</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">3</td>
                      <td className="px-4 py-4 text-center">
                        <CheckCircle className="w-5 h-5 text-green-500 inline-block" />
                      </td>
                      <td className="px-6 py-4 text-center text-sm font-semibold text-gray-900">8.5</td>
                    </tr>

                    {/* Row 3: Optimize Blog SEO */}
                    <tr className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 text-center">
                        <div>
                          <a href="#" className="text-sm font-medium text-blue-600 hover:text-blue-700">Optimize Blog SEO</a>
                          <div className="mt-1">
                            <span className="inline-block px-2 py-0.5 bg-orange-100 text-orange-700 text-xs font-semibold rounded">High</span>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">4</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">2</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">4</td>
                      <td className="px-4 py-4 text-center">
                        <CheckCircle className="w-5 h-5 text-green-500 inline-block" />
                      </td>
                      <td className="px-6 py-4 text-center text-sm font-semibold text-gray-900">8.0</td>
                    </tr>

                    {/* Row 4: Standardize Social Visuals */}
                    <tr className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 text-center">
                        <div>
                          <a href="#" className="text-sm font-medium text-blue-600 hover:text-blue-700">Standardize Social Visuals</a>
                          <div className="mt-1">
                            <span className="inline-block px-2 py-0.5 bg-yellow-100 text-yellow-700 text-xs font-semibold rounded">Medium</span>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">3</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">3</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">5</td>
                      <td className="px-4 py-4 text-center">
                        <CheckCircle className="w-5 h-5 text-green-500 inline-block" />
                      </td>
                      <td className="px-6 py-4 text-center text-sm font-semibold text-gray-900">7.8</td>
                    </tr>

                    {/* Row 5: Launch Testimonial Campaign */}
                    <tr className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 text-center">
                        <div>
                          <a href="#" className="text-sm font-medium text-blue-600 hover:text-blue-700">Launch Testimonial Campaign</a>
                          <div className="mt-1">
                            <span className="inline-block px-2 py-0.5 bg-yellow-100 text-yellow-700 text-xs font-semibold rounded">Medium</span>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">3</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">3</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">4</td>
                      <td className="px-4 py-4 text-center">
                        <CheckCircle className="w-5 h-5 text-green-500 inline-block" />
                      </td>
                      <td className="px-6 py-4 text-center text-sm font-semibold text-gray-900">7.5</td>
                    </tr>

                    {/* Row 6: Engage with Influencers */}
                    <tr className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 text-center">
                        <div>
                          <a href="#" className="text-sm font-medium text-blue-600 hover:text-blue-700">Engage with Influencers</a>
                          <div className="mt-1">
                            <span className="inline-block px-2 py-0.5 bg-green-100 text-green-700 text-xs font-semibold rounded">Low</span>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">2</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">4</td>
                      <td className="px-4 py-4 text-center text-sm text-gray-900">3</td>
                      <td className="px-4 py-4 text-center">
                        <X className="w-5 h-5 text-red-500 inline-block" />
                      </td>
                      <td className="px-6 py-4 text-center text-sm font-semibold text-gray-900">6.5</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
