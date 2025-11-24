import { useState } from "react";
import { motion } from "motion/react";
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

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "bg-green-100 text-green-700 border-green-200";
      case "paused":
        return "bg-yellow-100 text-yellow-700 border-yellow-200";
      case "completed":
        return "bg-blue-100 text-blue-700 border-blue-200";
      case "draft":
        return "bg-gray-100 text-gray-700 border-gray-200";
      default:
        return "bg-gray-100 text-gray-700 border-gray-200";
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
              <h1 className="text-4xl font-bold text-gray-900 mb-2">Campaigns</h1>
              <p className="text-gray-600">Manage and track your marketing campaigns</p>
            </div>
            <Button className="!bg-black hover:bg-gray-800 text-white shadow-lg shadow-black/20 transition-all hover:scale-105">
              <Plus className="w-4 h-4 mr-2" />
              Create Campaign
            </Button>
          </div>
        </motion.div>

        {/* Stats Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-3 gap-8 mb-8"
        >
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-black rounded-xl flex items-center justify-center shadow-lg shadow-black/20">
                <Target className="w-8 h-8 text-black" />
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
              <div className="w-12 h-12 bg-black rounded-xl flex items-center justify-center shadow-lg shadow-black/20">
                <Eye className="w-8 h-8 text-black" />
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
              <div className="w-12 h-12 bg-black rounded-xl flex items-center justify-center shadow-lg shadow-black/20">
                <Users className="w-8 h-8 text-black" />
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
          <div className="p-6 border-b border-gray-200 flex flex-col sm:flex-row gap-4 justify-between items-center bg-gray-50/50">
            <h2 className="text-xl font-bold text-gray-900 w-full sm:w-auto">All Campaigns</h2>
            
            <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
              {/* Search */}
              <div className="relative w-full sm:w-auto">
                <Input
                  type="text"
                  placeholder="Search campaigns..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="!pl-12 border-gray-300 focus:border-black h-10 w-full sm:w-64 bg-white"
                />
              </div>

              {/* Status Filter */}
              <div className="relative w-full sm:w-auto">
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="!pl-12 pr-10 py-2 border border-gray-300 rounded-lg focus:border-black focus:outline-none h-10 bg-white text-sm appearance-none cursor-pointer hover:border-gray-400 transition-colors w-full sm:w-40"
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
                        <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center flex-shrink-0">
                          {campaign.platform === "Social Media" ? (
                            <Users className="w-5 h-5 text-gray-600" />
                          ) : campaign.platform === "Email" ? (
                            <FileText className="w-5 h-5 text-gray-600" />
                          ) : (
                            <Megaphone className="w-5 h-5 text-gray-600" />
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
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${getStatusColor(
                          campaign.status
                        )}`}
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
                <Button className="!bg-black hover:bg-gray-800 text-white shadow-lg shadow-black/20">
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
