import { useState } from "react";
import { motion } from "motion/react";
import {
  TrendingUp,
  TrendingDown,
  Users,
  Eye,
  MousePointer,
  ArrowUpRight,
  ArrowDownRight,
  BarChart3,
  PieChart,
  Activity,
  Sparkles,
  LineChart,
  Clock,
  Target,
  Zap,
} from "lucide-react";

export function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState("7d");

  const metrics = [
    {
      label: "Total Views",
      value: "245.8K",
      change: "+12.5%",
      trend: "up" as const,
      icon: Eye,
      gradientFrom: "from-blue-500",
      gradientTo: "to-blue-600",
      shadowColor: "shadow-blue-500/20",
    },
    {
      label: "Unique Visitors",
      value: "89.2K",
      change: "+8.3%",
      trend: "up" as const,
      icon: Users,
      gradientFrom: "from-green-500",
      gradientTo: "to-green-600",
      shadowColor: "shadow-green-500/20",
    },
    {
      label: "Engagement Rate",
      value: "64.2%",
      change: "+5.7%",
      trend: "up" as const,
      icon: Activity,
      gradientFrom: "from-purple-500",
      gradientTo: "to-purple-600",
      shadowColor: "shadow-purple-500/20",
    },
    {
      label: "Conversion Rate",
      value: "3.8%",
      change: "-1.2%",
      trend: "down" as const,
      icon: MousePointer,
      gradientFrom: "from-orange-500",
      gradientTo: "to-orange-600",
      shadowColor: "shadow-orange-500/20",
    },
  ];

  const topContent = [
    { title: "Summer Product Launch Campaign", views: "45.2K", engagement: "78%", platform: "LinkedIn", rank: 1 },
    { title: "5 Tips for Better Productivity", views: "38.7K", engagement: "72%", platform: "Twitter", rank: 2 },
    { title: "Behind the Scenes: Team Retreat", views: "32.1K", engagement: "85%", platform: "Instagram", rank: 3 },
    { title: "Q4 Marketing Strategy Webinar", views: "28.9K", engagement: "68%", platform: "Facebook", rank: 4 },
  ];

  const platformStats = [
    { name: "LinkedIn", percentage: 35, color: "bg-blue-500", lightColor: "bg-blue-50", textColor: "text-blue-700" },
    { name: "Twitter", percentage: 28, color: "bg-sky-500", lightColor: "bg-sky-50", textColor: "text-sky-700" },
    { name: "Instagram", percentage: 22, color: "bg-pink-500", lightColor: "bg-pink-50", textColor: "text-pink-700" },
    { name: "Facebook", percentage: 15, color: "bg-indigo-500", lightColor: "bg-indigo-50", textColor: "text-indigo-700" },
  ];

  const weeklyData = [
    { day: "Mon", value: 65 },
    { day: "Tue", value: 78 },
    { day: "Wed", value: 72 },
    { day: "Thu", value: 85 },
    { day: "Fri", value: 92 },
    { day: "Sat", value: 68 },
    { day: "Sun", value: 55 },
  ];

  const hourlyData = Array.from({ length: 24 }, (_, i) => ({
    hour: i,
    value: Math.floor(Math.random() * 40) + 30,
  }));

  const deviceData = [
    { name: "Desktop", percentage: 45, color: "from-blue-500 to-blue-600", icon: "💻" },
    { name: "Mobile", percentage: 38, color: "from-green-500 to-green-600", icon: "📱" },
    { name: "Tablet", percentage: 17, color: "from-purple-500 to-purple-600", icon: "📱" },
  ];

  const engagementMetrics = [
    { label: "Avg. Session Duration", value: "4m 32s", change: "+18%", trend: "up" as const, icon: Clock },
    { label: "Pages per Session", value: "5.8", change: "+12%", trend: "up" as const, icon: Eye },
    { label: "Bounce Rate", value: "32.4%", change: "-8%", trend: "down" as const, icon: TrendingDown },
    { label: "Return Visitors", value: "42.1%", change: "+15%", trend: "up" as const, icon: Users },
  ];

  return (
    <div className="p-8">
      <div className="max-w-[1600px] mx-auto space-y-10">
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-2"
        >
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent mb-2">
              Analytics Dashboard
            </h1>
            <p className="text-gray-600 text-lg">Track performance and insights across all channels</p>
          </div>

          {/* Time Range Selector */}
          <div className="flex items-center gap-2 bg-white rounded-2xl border border-gray-200 p-1.5 shadow-sm">
            {[
              { value: "24h", label: "24 Hours" },
              { value: "7d", label: "7 Days" },
              { value: "30d", label: "30 Days" },
              { value: "90d", label: "90 Days" },
            ].map((range) => (
              <button
                key={range.value}
                onClick={() => setTimeRange(range.value)}
                className={`px-5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 ${timeRange === range.value
                    ? "bg-gradient-to-r from-gray-900 to-gray-800 text-black shadow-lg shadow-gray-900/20"
                    : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
                  }`}
              >
                {range.label}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-4 gap-8 mt-8">
          {metrics.map((metric, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 + index * 0.05 }}
              className="group relative bg-white rounded-3xl shadow-sm border border-gray-200 p-8 hover:shadow-xl hover:scale-[1.02] hover:border-gray-300 transition-all duration-300 min-h-[180px] flex flex-col"
            >
              <div className="flex items-start justify-between mb-auto">
                <div className={`w-14 h-14 bg-gradient-to-br ${metric.gradientFrom} ${metric.gradientTo} rounded-2xl flex items-center justify-center shadow-lg ${metric.shadowColor} group-hover:scale-110 transition-transform duration-300`}>
                  <metric.icon className="w-7 h-7 text-black" />
                </div>
                <div className={`flex items-center gap-1.5 text-xs font-bold px-4 py-1.5 rounded-full ${metric.trend === "up" ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
                  }`}>
                  {metric.trend === "up" ? (
                    <ArrowUpRight className="w-4 h-4" />
                  ) : (
                    <ArrowDownRight className="w-4 h-4" />
                  )}
                  {metric.change}
                </div>
              </div>
              <div className="mt-4">
                <p className="text-sm font-semibold text-gray-500 mb-2">{metric.label}</p>
                <p className="text-4xl font-bold text-gray-900">{metric.value}</p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Main Chart - Traffic Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-3xl shadow-lg border border-gray-200 p-12 mt-8"
        >
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Traffic Overview</h2>
              <p className="text-gray-500">Daily visitor trends for the selected period</p>
            </div>
            <div className="flex items-center gap-2 bg-gray-50 rounded-xl p-1.5 border border-gray-200">
              <button className="p-2.5 bg-white hover:bg-gray-100 rounded-lg transition-colors shadow-sm border border-gray-200">
                <LineChart className="w-5 h-5 text-gray-700" />
              </button>
              <button className="p-2.5 hover:bg-white rounded-lg transition-colors">
                <BarChart3 className="w-5 h-5 text-gray-500" />
              </button>
            </div>
          </div>

          {/* Bar Chart */}
          <div className="relative" style={{ height: '320px' }}>
            {/* Grid lines */}
            <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
              {[0, 1, 2, 3, 4].map((i) => (
                <div key={i} className="w-full border-t border-gray-100" />
              ))}
            </div>

            {/* Bars */}
            <div className="absolute inset-0 flex items-end justify-around gap-4 px-8 pb-12">
              {weeklyData.map((item, index) => (
                <div key={index} className="flex-1 flex flex-col items-center gap-4 h-full justify-end">
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height: `${item.value}%` }}
                    transition={{ delay: 0.5 + index * 0.1, duration: 0.8, ease: "easeOut" }}
                    className="w-full bg-gradient-to-t from-blue-600 via-blue-500 to-blue-400 rounded-t-2xl relative group cursor-pointer hover:from-blue-700 hover:via-blue-600 hover:to-blue-500 transition-all duration-300 shadow-lg shadow-blue-500/20 min-h-[40px]"
                  >
                    {/* Tooltip */}
                    <div className="absolute -top-16 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-all duration-200 bg-gray-900 text-white text-sm font-semibold px-4 py-2 rounded-xl whitespace-nowrap shadow-2xl z-30 pointer-events-none">
                      {item.value}K views
                      <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-gray-900 rotate-45"></div>
                    </div>
                  </motion.div>
                  <span className="text-sm font-bold text-gray-700">{item.day}</span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Two Column Grid */}
        <div className="grid grid-cols-2 gap-8 mt-8">
          {/* Platform Distribution */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-white rounded-3xl shadow-lg border border-gray-200 p-8"
          >
            <div className="flex items-center gap-3 mb-8">
              <div className="w-12 h-12 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-2xl flex items-center justify-center shadow-lg shadow-yellow-500/20">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Platform Distribution</h2>
                <p className="text-sm text-gray-500">Traffic breakdown by platform</p>
              </div>
            </div>

            <div className="space-y-7">
              {platformStats.map((platform, index) => (
                <div key={index}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className={`w-4 h-4 rounded-full ${platform.color} shadow-md`} />
                      <span className="text-base font-bold text-gray-800">{platform.name}</span>
                    </div>
                    <span className="text-lg font-bold text-gray-900">{platform.percentage}%</span>
                  </div>
                  <div className={`w-full ${platform.lightColor} rounded-full h-4 overflow-hidden border border-gray-200`}>
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${platform.percentage}%` }}
                      transition={{ delay: 0.6 + index * 0.1, duration: 1, ease: "easeOut" }}
                      className={`h-4 rounded-full ${platform.color} shadow-sm`}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Total Reach Card */}
            <div className="mt-10 pt-8 border-t-2 border-gray-100">
              <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-2xl p-6 border border-green-200">
                <p className="text-sm font-semibold text-green-700 mb-2">Total Reach</p>
                <p className="text-4xl font-bold text-gray-900 mb-4">245.8K</p>
                <div className="flex items-center gap-2 text-green-700">
                  <TrendingUp className="w-5 h-5" />
                  <span className="text-sm font-bold">+12.5% from last period</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Conversion Funnel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.45 }}
            className="bg-white rounded-3xl shadow-lg border border-gray-200 p-8"
          >
            <div className="flex items-center gap-3 mb-8">
              <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center shadow-lg shadow-green-500/20">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Conversion Funnel</h2>
                <p className="text-sm text-gray-500">User journey stages</p>
              </div>
            </div>

            <div className="space-y-5">
              {[
                { stage: "Visitors", count: "245.8K", percentage: 100, color: "from-blue-500 to-blue-600" },
                { stage: "Engaged", count: "157.8K", percentage: 64, color: "from-green-500 to-green-600" },
                { stage: "Leads", count: "45.2K", percentage: 18, color: "from-yellow-500 to-orange-500" },
                { stage: "Conversions", count: "9.3K", percentage: 4, color: "from-purple-500 to-purple-600" },
              ].map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 + index * 0.1 }}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-base font-bold text-gray-800">{item.stage}</span>
                    <span className="text-sm font-bold text-gray-600">{item.count}</span>
                  </div>
                  <div className="relative h-14 bg-gray-100 rounded-2xl overflow-hidden border border-gray-200">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${item.percentage}%` }}
                      transition={{ delay: 0.7 + index * 0.1, duration: 0.8 }}
                      className={`h-full bg-gradient-to-r ${item.color} flex items-center px-5 shadow-lg`}
                    >
                      <span className="text-sm font-bold text-white">{item.percentage}%</span>
                    </motion.div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>


        {/* Engagement Metrics - Full Width */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white rounded-3xl shadow-lg border border-gray-200 p-12 mt-8"
        >
          <div className="flex items-center gap-3 mb-8">
            <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-green-600 rounded-2xl flex items-center justify-center shadow-lg shadow-green-500/20">
              <TrendingUp className="w-8 h-8 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Key Performance Metrics</h2>
              <p className="text-gray-500">Essential engagement and behavior indicators</p>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-8">
            {engagementMetrics.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 + index * 0.1 }}
                className="flex flex-col p-6 bg-gradient-to-br from-gray-50 to-white rounded-2xl hover:shadow-lg transition-all duration-200 cursor-pointer border border-gray-100 hover:border-gray-200 group"
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 bg-white rounded-xl flex items-center justify-center shadow-sm border border-gray-200 group-hover:scale-110 transition-transform">
                    <item.icon className="w-5 h-5 text-gray-600" />
                  </div>
                  <div className={`flex items-center gap-1 text-xs font-bold px-3 py-1.5 rounded-full ${item.trend === "up" ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
                    }`}>
                    {item.trend === "up" ? (
                      <ArrowUpRight className="w-3.5 h-3.5" />
                    ) : (
                      <ArrowDownRight className="w-3.5 h-3.5" />
                    )}
                    {item.change}
                  </div>
                </div>
                <p className="text-sm font-semibold text-gray-500 mb-2">{item.label}</p>
                <p className="text-3xl font-bold text-gray-900">{item.value}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>


        {/* Top Performing Content Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.65 }}
          className="bg-white rounded-3xl shadow-lg border border-gray-200 overflow-hidden mt-8 mb-8"
        >
          <div className="p-12 border-b border-gray-200 bg-gradient-to-r from-gray-50 to-white">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Top Performing Content</h2>
            <p className="text-gray-600">Your best content from the selected period</p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b-2 border-gray-200">
                <tr>
                  <th className="px-8 py-6 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Rank
                  </th>
                  <th className="px-8 py-6 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Content
                  </th>
                  <th className="px-8 py-6 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Platform
                  </th>
                  <th className="px-8 py-6 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Views
                  </th>
                  <th className="px-8 py-6 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Engagement
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {topContent.map((item, index) => (
                  <motion.tr
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.75 + index * 0.1 }}
                    className="hover:bg-gray-50 transition-colors group"
                  >
                    <td className="px-8 py-6">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-base shadow-sm ${item.rank === 1 ? "bg-gradient-to-br from-yellow-400 to-yellow-500 text-black" :
                          item.rank === 2 ? "bg-gradient-to-br from-gray-300 to-gray-400 text-black" :
                            item.rank === 3 ? "bg-gradient-to-br from-orange-400 to-orange-500 text-black" :
                              "bg-gray-100 text-gray-600"
                        }`}>
                        {item.rank}
                      </div>
                    </td>
                    <td className="px-8 py-6">
                      <div className="flex items-center gap-3">
                        <div className="w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse" />
                        <span className="font-semibold text-gray-900">{item.title}</span>
                      </div>
                    </td>
                    <td className="px-8 py-6">
                      <span className="inline-flex items-center px-4 py-2 rounded-xl text-xs font-bold bg-gray-100 text-gray-700 border border-gray-200">
                        {item.platform}
                      </span>
                    </td>
                    <td className="px-8 py-6">
                      <div className="flex items-center gap-2">
                        <Eye className="w-4 h-4 text-gray-400" />
                        <span className="text-base font-bold text-gray-900">{item.views}</span>
                      </div>
                    </td>
                    <td className="px-8 py-6">
                      <div className="flex items-center gap-4">
                        <div className="flex-1 max-w-[160px]">
                          <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden border border-gray-200">
                            <div
                              className="h-3 rounded-full bg-gradient-to-r from-green-500 to-green-600 shadow-sm"
                              style={{ width: item.engagement }}
                            />
                          </div>
                        </div>
                        <span className="text-base font-bold text-gray-900 min-w-[3.5rem]">{item.engagement}</span>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
