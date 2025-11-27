import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Plus,
  Search,
  Filter,
  MoreVertical,
  Calendar,
  Target,
  TrendingUp,
  Users,
  DollarSign,
  Eye,
  Edit2,
  Trash2,
  Play,
  Pause,
  CheckCircle,
  FileText,
  Megaphone,
  Sparkles,
  Building2,
} from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";

interface Campaign {
  id: number;
  name: string;
  status: "active" | "paused" | "completed" | "draft";
  platform: string;
  impressions: string;
  clicks: string;
  conversions: string;
  startDate: string;
  endDate: string;
}

export function CampaignsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [topic, setTopic] = useState("");
  const [goal, setGoal] = useState("");
  const [industry, setIndustry] = useState("");
  const [duration, setDuration] = useState("2weeks");

  const campaigns: Campaign[] = [
    {
      id: 1,
      name: "Summer Product Launch",
      status: "active",
      platform: "Multi-Platform",
      impressions: "125.4K",
      clicks: "8,234",
      conversions: "456",
      startDate: "2024-11-01",
      endDate: "2024-12-31",
    },
    {
      id: 2,
      name: "Holiday Season Sale",
      status: "active",
      platform: "Social Media",
      impressions: "89.2K",
      clicks: "5,678",
      conversions: "312",
      startDate: "2024-11-15",
      endDate: "2024-12-25",
    },
    {
      id: 3,
      name: "Brand Awareness Q4",
      status: "paused",
      platform: "Display Ads",
      impressions: "45.6K",
      clicks: "2,345",
      conversions: "123",
      startDate: "2024-10-01",
      endDate: "2024-12-31",
    },
    {
      id: 4,
      name: "Email Marketing Campaign",
      status: "completed",
      platform: "Email",
      impressions: "32.1K",
      clicks: "4,567",
      conversions: "234",
      startDate: "2024-09-01",
      endDate: "2024-10-31",
    },
  ];

  const suggestedTopics = [
    "Product Launch Q1 2025",
    "Brand Awareness Drive",
    "Holiday Season Sale"
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return { bg: "rgba(16, 185, 129, 0.15)", text: "#059669", border: "#6EE7B7" };
      case "paused":
        return { bg: "rgba(245, 158, 11, 0.15)", text: "#D97706", border: "#FCD34D" };
      case "completed":
        return { bg: "rgba(59, 130, 246, 0.15)", text: "#2563EB", border: "#93C5FD" };
      case "draft":
        return { bg: "rgba(107, 114, 128, 0.15)", text: "#4B5563", border: "#D1D5DB" };
      default:
        return { bg: "rgba(107, 114, 128, 0.15)", text: "#4B5563", border: "#D1D5DB" };
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "active":
        return <Play className="w-3.5 h-3.5" />;
      case "paused":
        return <Pause className="w-3.5 h-3.5" />;
      case "completed":
        return <CheckCircle className="w-3.5 h-3.5" />;
      default:
        return null;
    }
  };

  const filteredCampaigns = campaigns.filter((campaign) => {
    const matchesSearch = campaign.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterStatus === "all" || campaign.status === filterStatus;
    return matchesSearch && matchesFilter;
  });

  const handleSubmit = () => {
    console.log({ topic, goal, industry, duration });
    setIsFormOpen(false);
    setTopic("");
    setGoal("");
    setIndustry("");
    setDuration("2weeks");
  };

  return (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">Campaigns</h1>
              <p className="text-gray-600">Manage and track your marketing campaigns</p>
            </div>
            <Button
              onClick={() => setIsFormOpen(!isFormOpen)}
              className="!bg-black hover:bg-gray-800 text-white shadow-lg shadow-black/20 transition-all hover:scale-105"
            >
              <Plus className="w-4 h-4 mr-2" />
              Create Campaign
            </Button>
          </div>
        </motion.div>

        {/* Inline Campaign Creation Form */}
        <AnimatePresence>
          {isFormOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="overflow-hidden mb-8"
            >
              <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
                <div className="space-y-6">
                  {/* Topic Input */}
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Campaign Topic
                    </label>
                    <Input
                      type="text"
                      placeholder="Enter your campaign topic..."
                      value={topic}
                      onChange={(e) => setTopic(e.target.value)}
                      className="w-full"
                    />
                  </div>

                  {/* AI Suggested Topics */}
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <Sparkles className="w-4 h-4" style={{ color: "#8B5CF6" }} />
                      <label className="text-sm font-semibold text-gray-700">
                        AI Suggested Topics
                      </label>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      {suggestedTopics.map((suggestedTopic, index) => (
                        <motion.button
                          key={index}
                          type="button"
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          onClick={() => setTopic(suggestedTopic)}
                          className="p-4 rounded-xl border-2 border-gray-200 hover:border-purple-300 bg-gradient-to-br from-purple-50 to-white text-sm font-medium text-gray-700 hover:text-purple-700 transition-all text-left"
                        >
                          {suggestedTopic}
                        </motion.button>
                      ))}
                    </div>
                  </div>

                  {/* Goal of Campaign */}
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      <div className="flex items-center gap-2">
                        <Target className="w-4 h-4" style={{ color: "#3B82F6" }} />
                        Goal of Campaign
                      </div>
                    </label>
                    <Input
                      type="text"
                      placeholder="e.g., Increase brand awareness, Generate leads..."
                      value={goal}
                      onChange={(e) => setGoal(e.target.value)}
                      className="w-full"
                    />
                  </div>

                  {/* Target Industry */}
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      <div className="flex items-center gap-2">
                        <Building2 className="w-4 h-4" style={{ color: "#10B981" }} />
                        Target Industry
                      </div>
                    </label>
                    <Input
                      type="text"
                      placeholder="e.g., Technology, Healthcare, Finance..."
                      value={industry}
                      onChange={(e) => setIndustry(e.target.value)}
                      className="w-full"
                    />
                  </div>

                  {/* Duration */}
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4" style={{ color: "#F59E0B" }} />
                        Duration
                      </div>
                    </label>
                    <select
                      value={duration}
                      onChange={(e) => setDuration(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:border-black focus:outline-none bg-white text-sm cursor-pointer hover:border-gray-400 transition-colors"
                    >
                      <option value="1week">1 Week</option>
                      <option value="2weeks">2 Weeks</option>
                      <option value="1month">1 Month</option>
                    </select>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex items-center justify-end gap-3 pt-6 border-t border-gray-200">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setIsFormOpen(false)}
                      className="px-6"
                    >
                      Cancel
                    </Button>
                    <Button
                      type="button"
                      onClick={handleSubmit}
                      className="!bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white px-6 shadow-lg shadow-purple-500/30"
                    >
                      Create Campaign
                    </Button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Stats Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-3 gap-8 mb-8"
        >
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center shadow-lg"
                style={{
                  background: "linear-gradient(135deg, #6366F1 0%, #4F46E5 100%)",
                  boxShadow: "0 10px 15px -3px rgba(99, 102, 241, 0.3)"
                }}
              >
                <Target className="w-6 h-6 text-white" />
              </div>
              <span className="text-xs font-medium px-2 py-1 bg-green-100 text-green-700 rounded-full flex items-center gap-1">
                <TrendingUp className="w-3 h-3" />
                +12%
              </span>
            </div>
            <p className="text-sm font-medium text-gray-500 mb-1">Active Campaigns</p>
            <p className="text-3xl font-bold text-gray-900">2</p>
          </div>

          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center shadow-lg"
                style={{
                  background: "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)",
                  boxShadow: "0 10px 15px -3px rgba(59, 130, 246, 0.3)"
                }}
              >
                <Eye className="w-6 h-6 text-white" />
              </div>
              <span className="text-xs font-medium px-2 py-1 bg-green-100 text-green-700 rounded-full flex items-center gap-1">
                <TrendingUp className="w-3 h-3" />
                +5.4%
              </span>
            </div>
            <p className="text-sm font-medium text-gray-500 mb-1">Total Impressions</p>
            <p className="text-3xl font-bold text-gray-900">292K</p>
          </div>

          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center shadow-lg"
                style={{
                  background: "linear-gradient(135deg, #10B981 0%, #059669 100%)",
                  boxShadow: "0 10px 15px -3px rgba(16, 185, 129, 0.3)"
                }}
              >
                <Users className="w-6 h-6 text-white" />
              </div>
              <span className="text-xs font-medium px-2 py-1 bg-green-100 text-green-700 rounded-full flex items-center gap-1">
                <TrendingUp className="w-3 h-3" />
                +8.2%
              </span>
            </div>
            <p className="text-sm font-medium text-gray-500 mb-1">Total Clicks</p>
            <p className="text-3xl font-bold text-gray-900">20.8K</p>
          </div>
        </motion.div>

        {/* Campaigns List Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden"
        >
          {/* Card Header with Search & Filter */}
          <div className="p-6 border-b border-gray-200 flex items-center justify-between gap-4 bg-gray-50/50">
            <h2 className="text-xl font-bold text-gray-900">All Campaigns</h2>

            <div className="flex items-center gap-3">
              {/* Search */}
              <div className="relative">
                <Input
                  type="text"
                  placeholder="Search campaigns..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="!pl-12 border-gray-300 focus:border-black h-10 w-64 bg-white"
                />
              </div>

              {/* Status Filter */}
              <div className="relative">
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="!pl-12 pr-10 py-2 border border-gray-300 rounded-lg focus:border-black focus:outline-none h-10 bg-white text-sm appearance-none cursor-pointer hover:border-gray-400 transition-colors w-40"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="paused">Paused</option>
                  <option value="completed">Completed</option>
                  <option value="draft">Draft</option>
                </select>
              </div>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Campaign Details
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Key Metrics
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Timeline
                  </th>
                  <th className="px-6 py-4 text-right text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {filteredCampaigns.map((campaign, index) => (
                  <motion.tr
                    key={campaign.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 * index }}
                    className="hover:bg-gray-50/80 transition-colors group"
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: "linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%)", border: "1px solid #E5E7EB" }}>
                          {campaign.platform === "Social Media" ? (
                            <Users className="w-5 h-5" style={{ color: "#EC4899" }} />
                          ) : campaign.platform === "Email" ? (
                            <FileText className="w-5 h-5" style={{ color: "#3B82F6" }} />
                          ) : (
                            <Megaphone className="w-5 h-5" style={{ color: "#8B5CF6" }} />
                          )}
                        </div>
                        <div>
                          <p className="font-semibold text-gray-900">{campaign.name}</p>
                          <p className="text-xs text-gray-500">{campaign.platform} • #{campaign.id}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border"
                        style={{
                          backgroundColor: getStatusColor(campaign.status).bg,
                          color: getStatusColor(campaign.status).text,
                          borderColor: getStatusColor(campaign.status).border,
                        }}
                      >
                        {getStatusIcon(campaign.status)}
                        {campaign.status.charAt(0).toUpperCase() + campaign.status.slice(1)}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex gap-4">
                        <div className="text-center">
                          <p className="text-xs text-gray-500 mb-0.5">Impr.</p>
                          <p className="text-sm font-semibold text-gray-900">{campaign.impressions}</p>
                        </div>
                        <div className="text-center">
                          <p className="text-xs text-gray-500 mb-0.5">Clicks</p>
                          <p className="text-sm font-semibold text-gray-900">{campaign.clicks}</p>
                        </div>
                        <div className="text-center">
                          <p className="text-xs text-gray-500 mb-0.5">Conv.</p>
                          <p className="text-sm font-semibold text-gray-900">{campaign.conversions}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2 text-xs text-gray-600 bg-gray-50 px-2 py-1 rounded border border-gray-200 w-fit">
                        <Calendar className="w-3.5 h-3.5" />
                        <span>
                          {new Date(campaign.startDate).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} - {new Date(campaign.endDate).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-8 w-8 p-0 border-gray-200 hover:border-gray-300 hover:bg-white"
                          title="View Details"
                        >
                          <Eye className="w-4 h-4 text-gray-600" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-8 w-8 p-0 border-gray-200 hover:border-gray-300 hover:bg-white"
                          title="Edit Campaign"
                        >
                          <Edit2 className="w-4 h-4 text-gray-600" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-8 w-8 p-0 border-red-100 text-red-600 hover:bg-red-50 hover:border-red-200"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          {filteredCampaigns.length === 0 && (
            <div className="text-center py-24 bg-gray-50/50">
              <div className="w-20 h-20 mx-auto bg-white rounded-full flex items-center justify-center mb-4 shadow-sm border border-gray-100">
                <Target className="w-10 h-10 text-gray-300" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No campaigns found</h3>
              <p className="text-gray-600 mb-6 max-w-sm mx-auto">
                {searchQuery
                  ? `No results found for "${searchQuery}". Try adjusting your search or filters.`
                  : "Get started by creating your first marketing campaign to track performance."}
              </p>
              {!searchQuery && (
                <Button
                  onClick={() => setIsFormOpen(true)}
                  className="!bg-black hover:bg-gray-800 text-white shadow-lg shadow-black/20"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Create Campaign
                </Button>
              )}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
