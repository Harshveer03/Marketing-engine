import React, { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  BarChart3,
  TrendingUp,
  AlertTriangle,
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

      {/* Performance Snapshot Section */}
      <div className="mb-8">
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
                  className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 hover:shadow-md transition-all w-[180px] lg:w-full lg:max-w-full flex-shrink-0 flex flex-col h-[220px] min-w-0 overflow-hidden"
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

                  {/* Metric Name - Fixed Height */}
                  <h3 className="text-sm font-semibold text-gray-700 mb-3 h-10 line-clamp-2 flex-shrink-0 overflow-hidden text-ellipsis">
                    {section.title}
                  </h3>

                  {/* Score - Fixed Height */}
                  <div className="mb-2 h-10 flex items-center flex-shrink-0">
                    <span className={`text-3xl font-bold ${colors.text} leading-none truncate`}>
                      {section.score}%
                    </span>
                  </div>

                  {/* Comparison - Fixed Height */}
                  <div className={`text-xs font-medium mb-3 h-5 flex items-center flex-shrink-0 ${getTrendColor()} truncate`}>
                    {getTrendText()}
                  </div>

                  {/* Description - Flexible but constrained */}
                  <p className="text-xs text-gray-500 line-clamp-2 mt-auto flex-shrink overflow-hidden text-ellipsis">
                    {section.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>

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
              className={`px-6 py-2.5 rounded-full font-medium transition-all ${
                activeTab === tab.id
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
              className={`px-6 py-2.5 rounded-full font-medium transition-all whitespace-nowrap ${
                selectedFilter === option
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
      <div className="-mx-8 px-2 mb-8">
        <div className="grid grid-cols-3 gap-3 w-full">
        {/* Left Column: Competitor Comparison Grid - 1/3 width */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="col-span-1 bg-white rounded-xl shadow-md border border-gray-200 p-8"
        >
          <h3 className="text-lg font-bold mb-4 text-gray-900">Competitor Comparison Grid</h3>
          
          {/* Comparison Table */}
          <div className="overflow-x-auto">
            <table className="text-sm">
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
                  <td className="py-3 pr-2 text-gray-700">Visibility</td>
                  <td className="text-center py-3 px-2 font-semibold text-gray-900">82%</td>
                  <td className="text-center py-3 px-2 text-gray-600">78%</td>
                  <td className="text-center py-3 px-2 text-gray-600">90%</td>
                  <td className="text-center py-3 px-2 text-gray-600">65%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 pr-2 text-gray-700">Authority</td>
                  <td className="text-center py-3 px-2 font-semibold text-gray-900">75%</td>
                  <td className="text-center py-3 px-2 text-gray-600">80%</td>
                  <td className="text-center py-3 px-2 text-gray-600">72%</td>
                  <td className="text-center py-3 px-2 text-gray-600">60%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 pr-2 text-gray-700">Content Coverage</td>
                  <td className="text-center py-3 px-2 font-semibold text-gray-900">88%</td>
                  <td className="text-center py-3 px-2 text-gray-600">85%</td>
                  <td className="text-center py-3 px-2 text-gray-600">75%</td>
                  <td className="text-center py-3 px-2 text-gray-600">55%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 pr-2 text-gray-700">Messaging Strength</td>
                  <td className="text-center py-3 px-2 font-semibold text-gray-900">70%</td>
                  <td className="text-center py-3 px-2 text-gray-600">72%</td>
                  <td className="text-center py-3 px-2 text-gray-600">68%</td>
                  <td className="text-center py-3 px-2 text-gray-600">50%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 pr-2 text-gray-700">Keyword Overlap</td>
                  <td className="text-center py-3 px-2 font-semibold text-gray-900">65%</td>
                  <td className="text-center py-3 px-2 text-gray-600">70%</td>
                  <td className="text-center py-3 px-2 text-gray-600">60%</td>
                  <td className="text-center py-3 px-2 text-gray-600">45%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 pr-2 text-gray-700">Engagement Footprint</td>
                  <td className="text-center py-3 px-2 font-semibold text-gray-900">78%</td>
                  <td className="text-center py-3 px-2 text-gray-600">75%</td>
                  <td className="text-center py-3 px-2 text-gray-600">85%</td>
                  <td className="text-center py-3 px-2 text-gray-600">62%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 pr-2 text-gray-700">Webwide Mentions</td>
                  <td className="text-center py-3 px-2 font-semibold text-gray-900">92%</td>
                  <td className="text-center py-3 px-2 text-gray-600">89%</td>
                  <td className="text-center py-3 px-2 text-gray-600">80%</td>
                  <td className="text-center py-3 px-2 text-gray-600">70%</td>
                </tr>
                <tr>
                  <td className="py-3 pr-2 text-gray-700">Trend Position</td>
                  <td className="text-center py-3 px-2 font-semibold text-green-600">Leader</td>
                  <td className="text-center py-3 px-2 text-gray-600">Follower</td>
                  <td className="text-center py-3 px-2 text-gray-600">Lagging</td>
                  <td className="text-center py-3 px-2 text-gray-600">Disruptor</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Summary Section */}
          <div className="mt-6 pt-4 border-t border-gray-200">
            <h4 className="text-sm font-semibold mb-3 text-gray-700">Summary</h4>
            <ul className="space-y-2 text-xs text-gray-600">
              <li>• Strong in brand mentions and content, leading in many areas.</li>
              <li>• Good visibility, slightly higher authority in key topics.</li>
              <li>• High visibility but weaker content coverage, focused on niche.</li>
              <li>• Emerging brand, significant growth potential in specific areas.</li>
            </ul>
          </div>
        </motion.div>

        {/* Right Column: Two Cards Stacked - 2/3 width */}
        <div className="col-span-3 space-y-6">
          {/* Card 2: Trend Position Mini Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white rounded-xl shadow-md border border-gray-200 p-8"
          >
            {/* Header Section */}
            <div className="mb-6">
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-gray-900 mb-1">Trend position panel</h3>
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
                className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm hover:shadow-md transition-all h-[200px] flex flex-col"
              >
                <h4 className="text-base font-bold text-gray-900 mb-2">Leader</h4>
                <p className="text-sm text-gray-600 mb-4 flex-shrink-0">
                  Sets the narrative and is referenced as a category-defining voice.
                </p>
                <div className="mt-auto space-y-2">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-600 rounded-full flex-shrink-0"></div>
                    <span className="text-sm text-gray-700">You</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-500">•</span>
                    <span className="text-sm text-gray-700">Comp A</span>
                  </div>
                </div>
              </motion.div>
              
              {/* Disruptor - Top Right */}
              <motion.div
                whileHover={{ scale: 1.02 }}
                className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm hover:shadow-md transition-all h-[200px] flex flex-col"
              >
                <h4 className="text-base font-bold text-gray-900 mb-2">Disruptor</h4>
                <p className="text-sm text-gray-600 mb-4 flex-shrink-0">
                  Introduces new angles and contrarian takes that reshape demand.
                </p>
                <div className="mt-auto space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-500">•</span>
                    <span className="text-sm text-gray-700">Comp B</span>
                  </div>
                </div>
              </motion.div>
              
              {/* Follower - Bottom Left */}
              <motion.div
                whileHover={{ scale: 1.02 }}
                className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm hover:shadow-md transition-all h-[200px] flex flex-col"
              >
                <h4 className="text-base font-bold text-gray-900 mb-2">Follower</h4>
                <p className="text-sm text-gray-600 mb-4 flex-shrink-0">
                  Joins trends late and mainly echoes existing narratives.
                </p>
                <div className="mt-auto space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-500">•</span>
                    <span className="text-sm text-gray-700">Comp C</span>
                  </div>
                </div>
              </motion.div>
              
              {/* Lagging - Bottom Right */}
              <motion.div
                whileHover={{ scale: 1.02 }}
                className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm hover:shadow-md transition-all h-[200px] flex flex-col"
              >
                <h4 className="text-base font-bold text-gray-900 mb-2">Lagging</h4>
                <p className="text-sm text-gray-600 mb-4 flex-shrink-0">
                  Rarely shows up in category conversations or emerging themes.
                </p>
                <div className="mt-auto space-y-2">
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
            className="bg-white rounded-xl shadow-md border border-gray-200 p-8"
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
