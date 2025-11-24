import { useState } from "react";
import { motion } from "motion/react";
import {
  Plus,
  Search,
  Filter,
  MoreVertical,
  Calendar,
  TrendingUp,
  MessageSquare,
  Share2,
  Heart,
  Eye,
  Edit2,
  Trash2,
  Clock,
  CheckCircle,
  FileText,
  Image as ImageIcon,
  Video,
  Linkedin,
  Twitter,
  Instagram,
  Facebook,
} from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";

interface Post {
  id: number;
  content: string;
  status: "published" | "scheduled" | "draft";
  platform: "linkedin" | "twitter" | "instagram" | "facebook";
  type: "text" | "image" | "video";
  likes: string;
  comments: string;
  shares: string;
  views: string;
  date: string;
  image?: string;
}

export function PostsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterStatus, setFilterStatus] = useState<string>("all");

  const posts: Post[] = [
    {
      id: 1,
      content: "Excited to announce our new product launch! 🚀 #ProductLaunch #Innovation",
      status: "published",
      platform: "linkedin",
      type: "image",
      likes: "1,245",
      comments: "89",
      shares: "45",
      views: "12.5K",
      date: "2024-11-24T10:00:00",
    },
    {
      id: 2,
      content: "5 tips for better productivity in 2024. Thread 🧵 👇",
      status: "published",
      platform: "twitter",
      type: "text",
      likes: "856",
      comments: "42",
      shares: "120",
      views: "8.2K",
      date: "2024-11-23T14:30:00",
    },
    {
      id: 3,
      content: "Behind the scenes at our annual team retreat! 🌲📸",
      status: "scheduled",
      platform: "instagram",
      type: "image",
      likes: "-",
      comments: "-",
      shares: "-",
      views: "-",
      date: "2024-11-26T09:00:00",
    },
    {
      id: 4,
      content: "Join us for a live Q&A session this Friday! 🎥",
      status: "draft",
      platform: "facebook",
      type: "video",
      likes: "-",
      comments: "-",
      shares: "-",
      views: "-",
      date: "2024-11-28T16:00:00",
    },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "published":
        return "bg-green-100 text-green-700 border-green-200";
      case "scheduled":
        return "bg-blue-100 text-blue-700 border-blue-200";
      case "draft":
        return "bg-gray-100 text-gray-700 border-gray-200";
      default:
        return "bg-gray-100 text-gray-700 border-gray-200";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "published":
        return <CheckCircle className="w-3.5 h-3.5" />;
      case "scheduled":
        return <Clock className="w-3.5 h-3.5" />;
      case "draft":
        return <FileText className="w-3.5 h-3.5" />;
      default:
        return null;
    }
  };

  const getPlatformIcon = (platform: string) => {
    switch (platform) {
      case "linkedin":
        return <Linkedin className="w-4 h-4 text-[#0077b5]" />;
      case "twitter":
        return <Twitter className="w-4 h-4 text-[#1da1f2]" />;
      case "instagram":
        return <Instagram className="w-4 h-4 text-[#e1306c]" />;
      case "facebook":
        return <Facebook className="w-4 h-4 text-[#1877f2]" />;
      default:
        return <Share2 className="w-4 h-4 text-gray-500" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "image":
        return <ImageIcon className="w-4 h-4 text-gray-500" />;
      case "video":
        return <Video className="w-4 h-4 text-gray-500" />;
      default:
        return <FileText className="w-4 h-4 text-gray-500" />;
    }
  };

  const filteredPosts = posts.filter((post) => {
    const matchesSearch = post.content.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterStatus === "all" || post.status === filterStatus;
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
              <h1 className="text-4xl font-bold text-gray-900 mb-2">Posts</h1>
              <p className="text-gray-600">Manage, schedule, and publish your content</p>
            </div>
            <Button className="!bg-black hover:bg-gray-800 text-white shadow-lg shadow-black/20 transition-all hover:scale-105">
              <Plus className="w-4 h-4 mr-2" />
              Create Post
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
              <div className="w-12 h-12 bg-white rounded-xl flex items-center justify-center shadow-lg shadow-black/20">
                <FileText className="w-6 h-6 text-black" />
              </div>
              <span className="text-xs font-medium px-2 py-1 bg-green-100 text-green-700 rounded-full flex items-center gap-1">
                <TrendingUp className="w-3 h-3" />
                +12%
              </span>
            </div>
            <p className="text-sm font-medium text-gray-500 mb-1">Total Posts</p>
            <p className="text-3xl font-bold text-gray-900">124</p>
          </div>

          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-white rounded-xl flex items-center justify-center shadow-lg shadow-black/20">
                <Clock className="w-6 h-6 text-black" />
              </div>
              <span className="text-xs font-medium px-2 py-1 bg-blue-100 text-blue-700 rounded-full flex items-center gap-1">
                <TrendingUp className="w-3 h-3" />
                +4
              </span>
            </div>
            <p className="text-sm font-medium text-gray-500 mb-1">Scheduled</p>
            <p className="text-3xl font-bold text-gray-900">8</p>
          </div>

          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-white rounded-xl flex items-center justify-center shadow-lg shadow-black/20">
                <Heart className="w-6 h-6 text-black" />
              </div>
              <span className="text-xs font-medium px-2 py-1 bg-green-100 text-green-700 rounded-full flex items-center gap-1">
                <TrendingUp className="w-3 h-3" />
                +24%
              </span>
            </div>
            <p className="text-sm font-medium text-gray-500 mb-1">Total Engagement</p>
            <p className="text-3xl font-bold text-gray-900">45.2K</p>
          </div>
        </motion.div>

        {/* Posts List Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden"
        >
          {/* Card Header with Search & Filter */}
          <div className="p-6 border-b border-gray-200 flex flex-col sm:flex-row gap-4 justify-between items-center bg-gray-50/50">
            <h2 className="text-xl font-bold text-gray-900 w-full sm:w-auto">Recent Posts</h2>
            
            <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
              {/* Search */}
              <div className="relative w-full sm:w-auto">
                <Input
                  type="text"
                  placeholder="Search posts..."
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
                  <option value="published">Published</option>
                  <option value="scheduled">Scheduled</option>
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
                    Content
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Platform
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Engagement
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-4 text-right text-xs font-bold text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {filteredPosts.map((post, index) => (
                  <motion.tr
                    key={post.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 * index }}
                    className="hover:bg-gray-50/80 transition-colors group"
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-start gap-3 max-w-md">
                        <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center flex-shrink-0 mt-1">
                          {getTypeIcon(post.type)}
                        </div>
                        <div>
                          <p className="font-medium text-gray-900 line-clamp-2">{post.content}</p>
                          <p className="text-xs text-gray-500 mt-1">ID: #{post.id}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${getStatusColor(
                          post.status
                        )}`}
                      >
                        {getStatusIcon(post.status)}
                        {post.status.charAt(0).toUpperCase() + post.status.slice(1)}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <div className="p-1.5 bg-gray-50 rounded-full border border-gray-100">
                          {getPlatformIcon(post.platform)}
                        </div>
                        <span className="text-sm text-gray-600 capitalize">{post.platform}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      {post.status === "published" ? (
                        <div className="flex gap-4">
                          <div className="flex items-center gap-1 text-gray-600" title="Likes">
                            <Heart className="w-3.5 h-3.5" />
                            <span className="text-xs font-medium">{post.likes}</span>
                          </div>
                          <div className="flex items-center gap-1 text-gray-600" title="Comments">
                            <MessageSquare className="w-3.5 h-3.5" />
                            <span className="text-xs font-medium">{post.comments}</span>
                          </div>
                          <div className="flex items-center gap-1 text-gray-600" title="Views">
                            <Eye className="w-3.5 h-3.5" />
                            <span className="text-xs font-medium">{post.views}</span>
                          </div>
                        </div>
                      ) : (
                        <span className="text-xs text-gray-400 italic">No data yet</span>
                      )}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2 text-xs text-gray-600 bg-gray-50 px-2 py-1 rounded border border-gray-200 w-fit">
                        <Calendar className="w-3.5 h-3.5" />
                        <span>
                          {new Date(post.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
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
                          title="Edit Post"
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

          {filteredPosts.length === 0 && (
            <div className="text-center py-24 bg-gray-50/50">
              <div className="w-20 h-20 mx-auto bg-white rounded-full flex items-center justify-center mb-4 shadow-sm border border-gray-100">
                <FileText className="w-10 h-10 text-gray-300" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No posts found</h3>
              <p className="text-gray-600 mb-6 max-w-sm mx-auto">
                {searchQuery
                  ? `No results found for "${searchQuery}". Try adjusting your search or filters.`
                  : "Start creating content to engage with your audience."}
              </p>
              {!searchQuery && (
                <Button className="!bg-black hover:bg-gray-800 text-white shadow-lg shadow-black/20">
                  <Plus className="w-4 h-4 mr-2" />
                  Create Post
                </Button>
              )}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
