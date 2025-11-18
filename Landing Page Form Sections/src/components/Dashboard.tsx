import { useState } from "react";
import { DashboardSidebar } from "./DashboardSidebar.tsx";
import {
  Plus,
  TrendingUp,
  CreditCard,
  Activity,
  BarChart3,
  FileText,
  Zap,
} from "lucide-react";
import { Button } from "./ui/button";
import { motion } from "motion/react";

export function Dashboard() {
  const [activeSection, setActiveSection] = useState("home");

  return (
    <div className="h-screen bg-gray-50 flex overflow-hidden">
      {/* Left Sidebar */}
      <DashboardSidebar
        activeSection={activeSection}
        onSectionChange={setActiveSection}
      />

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="p-8">
          {/* Greeting */}
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-bold text-gray-900 mb-8"
          >
            Good morning, User
          </motion.h1>

          {/* Action Buttons */}
          <div className="grid grid-cols-2 gap-6 mb-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
            >
              <Button
                style={{ backgroundColor: "#000000" }}
                className="w-full h-32 text-2xl font-semibold hover:bg-gray-800 text-white rounded-2xl shadow-lg hover:shadow-xl transition-all"
              >
                <FileText className="w-8 h-8 mr-3" />
                New Post
              </Button>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Button
                style={{ backgroundColor: "#000000" }}
                className="w-full h-32 text-2xl font-semibold hover:bg-gray-800 text-white rounded-2xl shadow-lg hover:shadow-xl transition-all"
              >
                <Zap className="w-8 h-8 mr-3" />
                New Campaign
              </Button>
            </motion.div>
          </div>

          {/* Brand Diagnostics & Credits */}
          <div className="grid grid-cols-2 gap-6 mb-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white p-8 rounded-2xl shadow-lg border border-gray-200"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-gray-900">
                  Brand Diagnostics
                </h3>
                <TrendingUp className="w-6 h-6 text-green-600" />
              </div>
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center">
                  <span className="text-2xl">✓</span>
                </div>
                <div>
                  <p className="text-3xl font-bold text-gray-900">85/100</p>
                  <p className="text-sm text-gray-500">Health Score</p>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-white p-8 rounded-2xl shadow-lg border border-gray-200"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-gray-900">
                  Credits Left
                </h3>
                <CreditCard className="w-6 h-6 text-blue-600" />
              </div>
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
                  <span className="text-2xl font-bold">25</span>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Available credits</p>
                  <Button className="mt-2 text-xs bg-black hover:bg-gray-800 text-white">
                    Buy More
                  </Button>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Engagement & Trends */}
          <div className="grid grid-cols-2 gap-6 mb-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-white p-8 rounded-2xl shadow-lg border border-gray-200"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-900">
                  Avg. Engagement
                </h3>
                <Activity className="w-6 h-6 text-purple-600" />
              </div>
              <div className="h-32 flex items-end gap-2">
                {[40, 65, 45, 80, 60, 90, 75].map((height, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-gray-200 rounded-t"
                    style={{ height: `${height}%` }}
                  />
                ))}
              </div>
              <p className="text-sm text-gray-500 mt-4">Last 7 days</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="bg-white p-8 rounded-2xl shadow-lg border border-gray-200"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-900">
                  View Industry Trends
                </h3>
                <BarChart3 className="w-6 h-6 text-orange-600" />
              </div>
              <div className="h-32 flex items-center justify-center">
                <svg className="w-full h-full" viewBox="0 0 200 100">
                  <polyline
                    points="0,80 40,60 80,70 120,40 160,50 200,20"
                    fill="none"
                    stroke="#000"
                    strokeWidth="3"
                  />
                </svg>
              </div>
              <Button className="w-full mt-4 bg-black hover:bg-gray-800 text-white">
                View Trends
              </Button>
            </motion.div>
          </div>

          {/* Analytics */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="bg-white p-8 rounded-2xl shadow-lg border border-gray-200"
          >
            <h3 className="text-2xl font-bold text-gray-900 mb-6">Analytics</h3>
            <div className="h-64 flex items-end gap-3">
              {[30, 50, 40, 70, 55, 85, 65, 90, 75, 60, 80, 70].map(
                (height, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-black rounded-t hover:bg-gray-800 transition-colors cursor-pointer"
                    style={{ height: `${height}%` }}
                  />
                )
              )}
            </div>
            <div className="flex justify-between mt-4 text-sm text-gray-500">
              <span>Jan</span>
              <span>Feb</span>
              <span>Mar</span>
              <span>Apr</span>
              <span>May</span>
              <span>Jun</span>
              <span>Jul</span>
              <span>Aug</span>
              <span>Sep</span>
              <span>Oct</span>
              <span>Nov</span>
              <span>Dec</span>
            </div>
          </motion.div>
        </div>
      </main>

      {/* Right Sidebar */}
      <aside className="w-80 bg-white border-l border-gray-200 p-6 overflow-auto">
        {/* Calendar */}
        <div className="mb-8">
          <h3 className="text-lg font-bold text-gray-900 mb-4">Calendar</h3>
          <div className="bg-gray-50 p-4 rounded-xl">
            <div className="text-center mb-4">
              <p className="font-semibold text-gray-900">November 2024</p>
            </div>
            <div className="grid grid-cols-7 gap-2 text-center text-xs mb-2">
              {["S", "M", "T", "W", "T", "F", "S"].map((day, i) => (
                <div key={i} className="font-semibold text-gray-600">
                  {day}
                </div>
              ))}
            </div>
            <div className="grid grid-cols-7 gap-2 text-center text-sm">
              {Array.from({ length: 30 }, (_, i) => i + 1).map((day) => (
                <button
                  key={day}
                  className={`h-8 rounded-lg hover:bg-gray-200 transition-colors ${
                    day === 18 ? "bg-black text-white" : "text-gray-700"
                  }`}
                >
                  {day}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Schedule */}
        <div>
          <h3 className="text-lg font-bold text-gray-900 mb-4">My Schedule</h3>
          <div className="space-y-3">
            {[
              { time: "10:00 AM", title: "Team Meeting" },
              { time: "2:00 PM", title: "Content Review" },
              { time: "4:30 PM", title: "Campaign Launch" },
            ].map((item, i) => (
              <div key={i} className="bg-gray-50 p-4 rounded-xl">
                <p className="text-xs text-gray-500">{item.time}</p>
                <p className="font-semibold text-gray-900">{item.title}</p>
              </div>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
