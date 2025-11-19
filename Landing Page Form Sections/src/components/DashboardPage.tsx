import { motion } from 'framer-motion';
import { 
  Plus, 
  Home, 
  Sparkles, 
  Megaphone, 
  FileText, 
  BarChart3, 
  Calendar as CalendarIcon, 
  Settings, 
  Crown, 
  HelpCircle,
  Bell,
  User,
  Check,
  ArrowUpRight
} from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface DashboardPageProps {
  userName?: string;
}

function DashboardPage({ userName = 'User' }: DashboardPageProps) {
  const currentHour = new Date().getHours();
  const greeting = currentHour < 12 ? 'Good morning' : currentHour < 18 ? 'Good afternoon' : 'Good evening';

  const sidebarItems = [
    { icon: Plus, label: '+', isButton: true },
    { icon: Home, label: 'Home', active: true },
    { icon: Sparkles, label: 'Brand' },
    { icon: Sparkles, label: 'Brand Diagnostics' },
    { icon: Megaphone, label: 'Campaigns' },
    { icon: FileText, label: 'Your Posts' },
    { icon: BarChart3, label: 'Analytics' },
    { icon: CalendarIcon, label: 'Calendar' },
    { icon: Settings, label: 'Settings' },
  ];

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Top Navigation */}
      <header className="border-b border-gray-200 bg-white sticky top-0 z-50">
        <div className="flex items-center justify-between px-8 py-3">
          <div className="text-xl font-bold">1SYX</div>
          <div className="flex items-center gap-4">
            <Button variant="outline" className="rounded-md px-6 py-1.5 text-sm border-gray-300">
              Upgrade
            </Button>
            <div className="flex items-center gap-3">
              <button className="relative">
                <Bell className="w-5 h-5 text-gray-700" />
              </button>
              <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center">
                <User className="w-5 h-5 text-gray-600" />
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-1">
        {/* Left Sidebar */}
        <aside className="w-56 border-r border-gray-200 bg-white flex flex-col">
          <div className="px-4 py-4 border-b border-gray-200">
            <div className="text-base font-semibold">1SYX</div>
          </div>
          
          <nav className="flex-1 px-3 py-4 space-y-0.5">
            {sidebarItems.map((item, index) => (
              <button
                key={index}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-sm transition-colors ${
                  item.active 
                    ? 'bg-gray-100 font-medium text-gray-900' 
                    : 'text-gray-700 hover:bg-gray-50'
                } ${item.isButton ? 'justify-center font-bold text-base mb-2' : ''}`}
              >
                <item.icon className="w-4 h-4" />
                {!item.isButton && <span className="text-sm">{item.label}</span>}
              </button>
            ))}
          </nav>

          <div className="px-3 pb-4 space-y-2">
            <Button className="w-full bg-black hover:bg-gray-800 text-white rounded-md py-5 flex items-center justify-center gap-2 text-sm font-medium">
              <Crown className="w-4 h-4" />
              <span>Upgrade to Pro</span>
            </Button>
            <button className="w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-sm text-gray-700 hover:bg-gray-50 transition-colors">
              <HelpCircle className="w-4 h-4" />
              <span>Need Help</span>
            </button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 px-10 py-6 overflow-auto bg-gray-50">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Enhanced Greeting Card - Full Width */}
            <Card className="p-6 rounded-xl border border-gray-200 bg-white shadow-sm mb-5">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-full bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center flex-shrink-0">
                  <User className="w-7 h-7 text-gray-700" />
                </div>
                <div className="flex-1">
                  <h1 className="text-2xl font-semibold text-gray-900 mb-1">
                    {greeting}, {userName} 👋
                  </h1>
                  <p className="text-sm text-gray-600">
                    {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })} • {new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            </Card>

            {/* New Post & New Campaign Cards */}
            <div className="grid grid-cols-2 gap-5 mb-5">
              {/* New Post Card */}
              <Card className="p-8 rounded-xl bg-black shadow-sm hover:shadow-md transition-shadow cursor-pointer group">
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-white" />
                  <h3 className="text-lg font-semibold text-white group-hover:text-gray-200 transition-colors">New Post</h3>
                </div>
              </Card>

              {/* New Campaign Card */}
              <Card className="p-8 rounded-xl bg-black shadow-sm hover:shadow-md transition-shadow cursor-pointer group">
                <div className="flex items-center gap-3">
                  <Sparkles className="w-5 h-5 text-white" />
                  <h3 className="text-lg font-semibold text-white group-hover:text-gray-200 transition-colors">New Campaign</h3>
                </div>
              </Card>
            </div>

            {/* Dashboard Cards Grid */}
            <div className="grid grid-cols-2 gap-5 mb-5">
              {/* Brand Diagnostics Card */}
              <Card className="p-5 rounded-xl border border-gray-200 bg-white shadow-sm">
                <h3 className="text-base font-semibold mb-6 text-gray-900">Brand Diagnostics</h3>
                <div className="flex items-center gap-3">
                  <div className="w-7 h-7 rounded-full bg-black flex items-center justify-center flex-shrink-0">
                    <Check className="w-4 h-4 text-white" />
                  </div>
                  <div className="flex-1 h-1.5 bg-gray-200 rounded-full">
                    <div className="w-3/4 h-full bg-gray-400 rounded-full"></div>
                  </div>
                </div>
              </Card>

              {/* Credits Left Card */}
              <Card className="p-5 rounded-xl border border-gray-200 bg-white shadow-sm">
                <h3 className="text-base font-semibold mb-6 text-gray-900">Credits Left</h3>
                <div className="flex items-center gap-3">
                  <div className="text-2xl font-bold border border-gray-300 rounded-lg px-3 py-1.5">
                    25
                  </div>
                  <div className="flex-1 h-1.5 bg-gray-200 rounded-full">
                    <div className="w-1/4 h-full bg-gray-400 rounded-full"></div>
                  </div>
                </div>
              </Card>

              {/* Total Engagement Card */}
              <Card className="p-5 rounded-xl border border-gray-200 bg-white shadow-sm">
                <h3 className="text-base font-semibold mb-4 text-gray-900">Total Engagement</h3>
                <div className="h-28 flex items-end">
                  <svg className="w-full h-full" viewBox="0 0 300 100" preserveAspectRatio="none">
                    <path
                      d="M 0 80 Q 75 60 150 70 T 300 50"
                      fill="none"
                      stroke="#d1d5db"
                      strokeWidth="2"
                    />
                  </svg>
                </div>
              </Card>

              {/* Trends Card */}
              <Card className="p-5 rounded-xl border border-gray-200 bg-white shadow-sm">
                <h3 className="text-base font-semibold mb-4 text-gray-900">Trends</h3>
                <div className="h-28 flex items-end relative">
                  <svg className="w-full h-full" viewBox="0 0 300 100" preserveAspectRatio="none">
                    <path
                      d="M 0 90 L 60 70 L 120 80 L 180 50 L 240 40 L 300 20"
                      fill="none"
                      stroke="#d1d5db"
                      strokeWidth="2"
                    />
                  </svg>
                  <ArrowUpRight className="absolute top-0 right-0 w-10 h-10 text-gray-300" />
                </div>
              </Card>
            </div>

            {/* Analytics Card - Full Width */}
            <Card className="p-5 rounded-xl border border-gray-200 bg-white shadow-sm">
              <h3 className="text-base font-semibold mb-4 text-gray-900">Analytics</h3>
              <div className="h-40">
                <svg className="w-full h-full" viewBox="0 0 800 200" preserveAspectRatio="none">
                  <path
                    d="M 0 150 Q 100 120 200 130 T 400 100 T 600 80 T 800 60"
                    fill="url(#gradient)"
                    stroke="#d1d5db"
                    strokeWidth="2"
                  />
                  <defs>
                    <linearGradient id="gradient" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#e5e7eb" stopOpacity="0.5" />
                      <stop offset="100%" stopColor="#e5e7eb" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
            </Card>
          </motion.div>
        </main>

        {/* Right Sidebar */}
        <aside className="w-[28rem] border-l border-gray-200 bg-white px-5 py-6 space-y-5">
          {/* Calendar Card */}
          <Card className="p-5 rounded-xl border border-gray-200 bg-white shadow-sm">
            <h3 className="text-base font-semibold mb-4 text-gray-900">Calendar</h3>
            <div className="grid grid-cols-7 gap-1.5">
              {Array.from({ length: 35 }).map((_, i) => (
                <div
                  key={i}
                  className="aspect-square bg-gray-100 rounded"
                ></div>
              ))}
            </div>
          </Card>

          {/* Coming Soon Card */}
          <Card className="p-5 rounded-xl border border-gray-200 bg-white shadow-sm h-20 flex items-center justify-center">
            <p className="text-gray-400 text-sm">Coming Soon</p>
          </Card>

          {/* My Schedule Card */}
          <Card className="p-5 rounded-xl border border-gray-200 bg-white shadow-sm">
            <h3 className="text-base font-semibold mb-4 text-gray-900">My Schedule</h3>
            <div className="space-y-2.5">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="h-2.5 bg-gray-200 rounded-full"></div>
              ))}
            </div>
          </Card>
        </aside>
      </div>
    </div>
  );
}

export default DashboardPage;
