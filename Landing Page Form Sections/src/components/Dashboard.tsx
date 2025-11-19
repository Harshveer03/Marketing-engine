import { useState } from "react";
import { DashboardSidebar } from "./DashboardSidebar.tsx";
import { DashboardHeader } from "./DashboardHeader.tsx";
import { Footer } from "./Footer";
import Calendar from "react-calendar";
import "react-calendar/dist/Calendar.css";
import {
  TrendingUp,
  CreditCard,
  Activity,
  BarChart3,
  FileText,
  Zap,
} from "lucide-react";
import { Button } from "./ui/button";
import { motion } from "motion/react";

interface DashboardProps {
  onBackToLanding?: () => void;
}

export function Dashboard({ onBackToLanding }: DashboardProps) {
  const [activeSection, setActiveSection] = useState("home");
  const [selectedDate, setSelectedDate] = useState(new Date());

  return (
    <div className="h-screen bg-gray-50 flex flex-col overflow-hidden">
      {/* Dashboard Header */}
      <DashboardHeader creditsLeft={25} onBackToLanding={onBackToLanding} />

      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <DashboardSidebar
          activeSection={activeSection}
          onSectionChange={setActiveSection}
        />

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          <div className="p-8">
            {/* Enhanced Greeting Card - Full Width */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="relative px-8 py-12 rounded-xl shadow-sm border border-gray-200 mb-6 overflow-hidden"
            >
              {/* Background Image */}
              <div 
                className="absolute inset-0 bg-cover bg-center"
                style={{
                  backgroundImage: "url('https://images.unsplash.com/photo-1557683316-973673baf926?w=1200&h=400&fit=crop')",
                }}
              />
              {/* Overlay for readability */}
              <div className="absolute inset-0 bg-gradient-to-r from-blue-900/90 via-purple-900/85 to-pink-900/90" />
              
              {/* Content */}
              <div className="relative z-10 flex items-center gap-5">
                {/* Left: Avatar */}
                <div className="w-14 h-14 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center flex-shrink-0 border border-white/30">
                  <span className="text-2xl">👤</span>
                </div>
                
                {/* Center: Greeting & Welcome Message */}
                <div className="flex-1">
                  <h1 className="text-2xl font-semibold text-white mb-0.5">
                    Good morning, User 👋
                  </h1>
                  <p className="text-white/80 text-sm">
                    Welcome back! Ready to create something amazing today?
                  </p>
                </div>

                {/* Right: Decorative Element */}
                <div className="flex-shrink-0">
                  <div className="w-10 h-10 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center border border-white/30">
                    <span className="text-xl">✨</span>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Action Buttons */}
            <div className="grid grid-cols-2 gap-8 mb-8">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
              >
                <Button
                  style={{ backgroundColor: "#000000" }}
                  className="w-full h-16 text-2xl font-semibold hover:bg-gray-800 text-white rounded-2xl shadow-lg hover:shadow-xl transition-all"
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
                  className="w-full h-16 text-2xl font-semibold hover:bg-gray-800 text-white rounded-2xl shadow-lg hover:shadow-xl transition-all"
                >
                  <Zap className="w-8 h-8 mr-3" />
                  New Campaign
                </Button>
              </motion.div>
            </div>

            {/* Brand Diagnostics & Credits */}
            <div className="flex gap-8 mb-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                style={{ width: '70%' }}
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
                    <p className="text-sm text-gray-500">Score</p>
                  </div>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                style={{ width: '30%' }}
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
            <div className="grid grid-cols-2 gap-8 mb-8">
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
              <h3 className="text-2xl font-bold text-gray-900 mb-6">
                Analytics
              </h3>
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
        <aside className="w-64 bg-white border-l border-gray-200 p-6 overflow-auto">
          {/* Calendar */}
          <div className="mb-10">
            <h3 className="text-lg font-bold text-gray-900 mb-4">Calendar</h3>
            <div className="calendar-widget">
              <Calendar
                onChange={(value) => setSelectedDate(value as Date)}
                value={selectedDate}
                className="border-0 rounded-xl shadow-sm"
              />
            </div>
          </div>

          {/* Schedule */}
          <div>
            <h3 className="text-lg font-bold text-gray-900 mb-4">
              My Schedule
            </h3>
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
      
      {/* Footer */}
      <Footer />
    </div>
  );
}
