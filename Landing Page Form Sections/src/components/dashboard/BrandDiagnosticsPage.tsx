import { useState } from "react";
import { motion } from "motion/react";
import { BarChart3, Zap, TrendingUp, AlertTriangle } from "lucide-react";

export function BrandDiagnosticsPage() {
  const [primaryBrand] = useState("Nexus Tech");
  const [competitor] = useState("Vortex Systems");

  return (
    <div className="p-6 space-y-4">
      {/* Top Section - 2 Column Grid (30% / 70%) */}
      <div className="grid gap-4 items-start" style={{ gridTemplateColumns: "30% 1fr" }}>
        {/* Left Column - Stacked Cards */}
        <div className="flex flex-col gap-4">
          {/* Overall Health Score Card */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-black rounded-xl shadow-lg p-6 text-white min-h-[200px] flex flex-col items-center justify-center"
          >
            <h3 className="text-lg font-semibold mb-6">Overall Score</h3>
            
            {/* Circular Progress */}
            <div className="relative w-32 h-32">
              <svg className="w-full h-full transform -rotate-90">
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="rgba(207, 21, 21, 0.1)"
                  strokeWidth="8"
                  fill="none"
                />
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="#3b82f6"
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray="351.86"
                  strokeDashoffset="87.96"
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-3xl font-bold">75%</span>
              </div>
            </div>
          </motion.div>

          {/* Clarity, Specificity & Relevance Card */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-5"
          >
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                <Zap className="w-4 h-4 text-blue-600" />
              </div>
              <h4 className="text-sm font-bold text-gray-900">Clarity, Specificity & Relevance</h4>
            </div>

            <div className="space-y-3 mb-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-600">Clarity of Value Prop</span>
                  <span className="font-semibold">85%</span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full">
                  <div className="h-full bg-blue-500 rounded-full" style={{ width: "85%" }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-600">Specificity of Claims</span>
                  <span className="font-semibold">72%</span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full">
                  <div className="h-full bg-blue-400 rounded-full" style={{ width: "72%" }}></div>
                </div>
              </div>
            </div>

            <p className="text-xs text-gray-500 leading-relaxed">
              <span className="font-semibold text-gray-700">Insight:</span> While clarity is high, claims need more specific data points.
            </p>
          </motion.div>
        </div>

        {/* Right Column - Multi-Dimensional Diagnosis (Full Height) */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col min-h-[500px]"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold text-gray-900">Multi-Dimensional Diagnosis</h3>
            <div className="flex items-center gap-4 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                <span className="text-gray-600">{primaryBrand}</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-pink-500"></div>
                <span className="text-gray-600">{competitor}</span>
              </div>
            </div>
          </div>

          <div className="flex-1 flex items-center justify-center bg-gray-50 rounded-lg">
            <div className="text-center">
              <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-2" />
              <p className="text-sm text-gray-500">Radar Chart Visualization</p>
              <p className="text-xs text-gray-400">Clarity • Specificity • Relevance • Influence</p>
              <p className="text-xs text-gray-400">Messaging Consistency • Posting Consistency • Gap Security</p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Bottom Section - 3 Column Grid */}
      <div className="grid grid-cols-3 gap-4">
        {/* GAP Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-xl shadow-sm border border-gray-200 p-5"
        >
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 bg-red-100 rounded-lg flex items-center justify-center">
              <AlertTriangle className="w-4 h-4 text-red-600" />
            </div>
            <h4 className="text-sm font-bold text-gray-900">Severity of GAP</h4>
          </div>
          <div className="h-32 flex items-center justify-center text-gray-400 text-sm">
            Content placeholder
          </div>
        </motion.div>

        {/* Trends Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white rounded-xl shadow-sm border border-gray-200 p-5"
        >
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-4 h-4 text-green-600" />
            </div>
            <h4 className="text-sm font-bold text-gray-900">Trends</h4>
          </div>
          <div className="h-32 flex items-center justify-center text-gray-400 text-sm">
            Content placeholder
          </div>
        </motion.div>

        {/* Consistency Engine Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white rounded-xl shadow-sm border border-gray-200 p-5"
        >
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 bg-orange-100 rounded-lg flex items-center justify-center">
              <BarChart3 className="w-4 h-4 text-orange-600" />
            </div>
            <h4 className="text-sm font-bold text-gray-900">Consistency Engine</h4>
          </div>

          <div className="flex gap-4 mb-4">
            <div className="flex-1 text-center">
              <div className="text-3xl font-bold text-indigo-600 mb-1">88%</div>
              <div className="text-xs text-gray-500">Messaging</div>
            </div>
            <div className="flex-1 text-center">
              <div className="text-3xl font-bold text-indigo-600 mb-1">95%</div>
              <div className="text-xs text-gray-500">Posting Freq</div>
            </div>
          </div>

          <div className="bg-green-50 border border-green-200 rounded-lg p-2 flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-white text-xs">✓</span>
            </div>
            <p className="text-xs text-green-700 font-medium">
              Outperforming competitor by +18%
            </p>
          </div>
        </motion.div>
      </div>

      {/* Clarity Section */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Clarity</h2>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <p className="text-sm text-gray-600 mb-6">Measures how clearly your brand communicates its value proposition and key messages to the target audience.</p>
          
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Brand Comparison</h4>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-600">{primaryBrand}</span>
                    <span className="font-semibold text-blue-600">85%</span>
                  </div>
                  <div className="h-3 bg-gray-100 rounded-full">
                    <div className="h-full bg-blue-500 rounded-full" style={{ width: "85%" }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-600">{competitor}</span>
                    <span className="font-semibold text-pink-600">78%</span>
                  </div>
                  <div className="h-3 bg-gray-100 rounded-full">
                    <div className="h-full bg-pink-500 rounded-full" style={{ width: "78%" }}></div>
                  </div>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Clarity Breakdown</h4>
              <svg viewBox="0 0 200 120" className="w-full">
                <rect x="20" y="90" width="30" height="30" fill="#3b82f6" />
                <rect x="60" y="60" width="30" height="60" fill="#3b82f6" />
                <rect x="100" y="40" width="30" height="80" fill="#3b82f6" />
                <rect x="140" y="70" width="30" height="50" fill="#3b82f6" />
                <text x="35" y="110" textAnchor="middle" fontSize="10" fill="#6b7280">Msg</text>
                <text x="75" y="110" textAnchor="middle" fontSize="10" fill="#6b7280">CTA</text>
                <text x="115" y="110" textAnchor="middle" fontSize="10" fill="#6b7280">Value</text>
                <text x="155" y="110" textAnchor="middle" fontSize="10" fill="#6b7280">Tone</text>
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Specificity Section */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Specificity</h2>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <p className="text-sm text-gray-600 mb-6">Evaluates how specific and concrete your brand claims and messaging are, with data-backed statements.</p>
          
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">72%</div>
              <div className="text-xs text-gray-600 mt-1">Specificity Score</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">+12%</div>
              <div className="text-xs text-gray-600 mt-1">vs Competitor</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">68</div>
              <div className="text-xs text-gray-600 mt-1">Data Points Used</div>
            </div>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Claim Types Distribution</h4>
            <div className="flex gap-2 h-8">
              <div className="bg-blue-500 rounded" style={{ width: "45%" }} title="Quantified Claims"></div>
              <div className="bg-blue-400 rounded" style={{ width: "30%" }} title="Comparative Claims"></div>
              <div className="bg-blue-300 rounded" style={{ width: "25%" }} title="General Claims"></div>
            </div>
            <div className="flex justify-between text-xs text-gray-600 mt-2">
              <span>Quantified (45%)</span>
              <span>Comparative (30%)</span>
              <span>General (25%)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Relevance Section */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Relevance</h2>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <p className="text-sm text-gray-600 mb-6">Assesses how well your content aligns with audience interests, needs, and current market trends.</p>
          
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Audience Alignment</h4>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">Target Demographics</span>
                  <span className="text-xs font-semibold">92%</span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full">
                  <div className="h-full bg-green-500 rounded-full" style={{ width: "92%" }}></div>
                </div>
                
                <div className="flex items-center justify-between mt-3">
                  <span className="text-xs text-gray-600">Pain Points Addressed</span>
                  <span className="text-xs font-semibold">78%</span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full">
                  <div className="h-full bg-blue-500 rounded-full" style={{ width: "78%" }}></div>
                </div>
                
                <div className="flex items-center justify-between mt-3">
                  <span className="text-xs text-gray-600">Trend Alignment</span>
                  <span className="text-xs font-semibold">85%</span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full">
                  <div className="h-full bg-purple-500 rounded-full" style={{ width: "85%" }}></div>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Relevance Over Time</h4>
              <svg viewBox="0 0 200 100" className="w-full">
                <polyline points="10,80 50,60 90,45 130,50 170,35" fill="none" stroke="#3b82f6" strokeWidth="2" />
                <polyline points="10,85 50,75 90,70 130,72 170,65" fill="none" stroke="#ec4899" strokeWidth="2" strokeDasharray="4" />
                <line x1="10" y1="90" x2="170" y2="90" stroke="#e5e7eb" strokeWidth="1" />
                <text x="10" y="100" fontSize="8" fill="#6b7280">Jan</text>
                <text x="90" y="100" fontSize="8" fill="#6b7280">Mar</text>
                <text x="170" y="100" fontSize="8" fill="#6b7280">May</text>
              </svg>
              <div className="flex gap-4 text-xs mt-2">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-0.5 bg-blue-500"></div>
                  <span className="text-gray-600">Your Brand</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-0.5 bg-pink-500"></div>
                  <span className="text-gray-600">Competitor</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* GAP Section */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">GAP</h2>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <p className="text-sm text-gray-600 mb-6">Identifies gaps between your brand positioning and competitor strengths, highlighting areas for improvement.</p>
          
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-4">Gap Analysis</h4>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs mb-2">
                    <span className="text-gray-600">Innovation</span>
                    <span className="text-red-600 font-semibold">-15%</span>
                  </div>
                  <div className="flex gap-1">
                    <div className="flex-1 h-3 bg-blue-500 rounded-l" style={{ width: "70%" }}></div>
                    <div className="flex-1 h-3 bg-red-100 rounded-r" style={{ width: "30%" }}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-xs mb-2">
                    <span className="text-gray-600">Customer Service</span>
                    <span className="text-green-600 font-semibold">+8%</span>
                  </div>
                  <div className="flex gap-1">
                    <div className="flex-1 h-3 bg-blue-500 rounded" style={{ width: "88%" }}></div>
                    <div className="flex-1 h-3 bg-gray-100 rounded" style={{ width: "12%" }}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-xs mb-2">
                    <span className="text-gray-600">Market Reach</span>
                    <span className="text-red-600 font-semibold">-22%</span>
                  </div>
                  <div className="flex gap-1">
                    <div className="flex-1 h-3 bg-blue-500 rounded-l" style={{ width: "58%" }}></div>
                    <div className="flex-1 h-3 bg-red-100 rounded-r" style={{ width: "42%" }}></div>
                  </div>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-4">Priority Areas</h4>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 bg-red-50 rounded-lg border border-red-200">
                  <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0" />
                  <div>
                    <div className="text-xs font-semibold text-gray-900">Market Reach</div>
                    <div className="text-xs text-gray-600">Critical gap requiring immediate attention</div>
                  </div>
                </div>
                
                <div className="flex items-center gap-3 p-3 bg-orange-50 rounded-lg border border-orange-200">
                  <AlertTriangle className="w-5 h-5 text-orange-600 flex-shrink-0" />
                  <div>
                    <div className="text-xs font-semibold text-gray-900">Innovation</div>
                    <div className="text-xs text-gray-600">Moderate gap, monitor closely</div>
                  </div>
                </div>
                
                <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg border border-green-200">
                  <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white text-xs">✓</span>
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-gray-900">Customer Service</div>
                    <div className="text-xs text-gray-600">Leading position maintained</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Message Consistency Section */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Message Consistency</h2>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <p className="text-sm text-gray-600 mb-6">Tracks how consistently your brand messaging is maintained across all channels and touchpoints.</p>
          
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-4">Channel Consistency</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">Website</span>
                  <span className="text-xs font-semibold text-green-600">95%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">Social Media</span>
                  <span className="text-xs font-semibold text-blue-600">88%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">Email</span>
                  <span className="text-xs font-semibold text-blue-600">92%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">Advertising</span>
                  <span className="text-xs font-semibold text-yellow-600">78%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">PR/Media</span>
                  <span className="text-xs font-semibold text-blue-600">85%</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-4">Consistency Score Trend</h4>
              <div className="bg-gradient-to-r from-blue-50 to-green-50 rounded-lg p-4 mb-4">
                <div className="text-3xl font-bold text-blue-600">88%</div>
                <div className="text-xs text-gray-600">Overall Consistency</div>
                <div className="text-xs text-green-600 mt-1">↑ +5% from last month</div>
              </div>
              
              <svg viewBox="0 0 200 80" className="w-full">
                <polyline points="10,60 40,55 70,50 100,45 130,42 160,38 190,35" fill="none" stroke="#3b82f6" strokeWidth="2" />
                <circle cx="190" cy="35" r="3" fill="#3b82f6" />
                <line x1="10" y1="70" x2="190" y2="70" stroke="#e5e7eb" strokeWidth="1" />
                <text x="10" y="80" fontSize="8" fill="#6b7280">Jan</text>
                <text x="100" y="80" fontSize="8" fill="#6b7280">Mar</text>
                <text x="190" y="80" fontSize="8" fill="#6b7280">May</text>
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Post Consistency Section */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Post Consistency</h2>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <p className="text-sm text-gray-600 mb-6">Monitors your posting frequency and regularity across platforms to maintain audience engagement.</p>
          
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-4">Posting Frequency</h4>
              <div className="grid grid-cols-7 gap-1 mb-4">
                {[...Array(35)].map((_, i) => {
                  const intensity = Math.floor(Math.random() * 4);
                  const colors = ['bg-gray-100', 'bg-green-200', 'bg-green-400', 'bg-green-600'];
                  return <div key={i} className={`h-6 rounded ${colors[intensity]}`}></div>;
                })}
              </div>
              <div className="flex justify-between text-xs text-gray-500">
                <span>Less</span>
                <span>More Active</span>
              </div>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-4">Platform Activity</h4>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-600">LinkedIn</span>
                    <span className="font-semibold">5 posts/week</span>
                  </div>
                  <div className="h-2 bg-gray-100 rounded-full">
                    <div className="h-full bg-blue-500 rounded-full" style={{ width: "95%" }}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-600">Twitter</span>
                    <span className="font-semibold">12 posts/week</span>
                  </div>
                  <div className="h-2 bg-gray-100 rounded-full">
                    <div className="h-full bg-blue-500 rounded-full" style={{ width: "85%" }}></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-600">Instagram</span>
                    <span className="font-semibold">4 posts/week</span>
                  </div>
                  <div className="h-2 bg-gray-100 rounded-full">
                    <div className="h-full bg-blue-500 rounded-full" style={{ width: "80%" }}></div>
                  </div>
                </div>
                
                <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                  <div className="text-xs font-semibold text-gray-900">Consistency Score: 95%</div>
                  <div className="text-xs text-gray-600 mt-1">Excellent posting regularity maintained</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Trend Section */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Trend</h2>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <p className="text-sm text-gray-600 mb-6">Analyzes performance trends over time to identify growth patterns and areas requiring attention.</p>
          
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <TrendingUp className="w-6 h-6 text-green-600 mx-auto mb-2" />
              <div className="text-2xl font-bold text-green-600">+24%</div>
              <div className="text-xs text-gray-600 mt-1">Engagement Growth</div>
            </div>
            <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
              <TrendingUp className="w-6 h-6 text-blue-600 mx-auto mb-2" />
              <div className="text-2xl font-bold text-blue-600">+18%</div>
              <div className="text-xs text-gray-600 mt-1">Brand Awareness</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
              <TrendingUp className="w-6 h-6 text-purple-600 mx-auto mb-2" />
              <div className="text-2xl font-bold text-purple-600">+31%</div>
              <div className="text-xs text-gray-600 mt-1">Sentiment Score</div>
            </div>
          </div>
          
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-4">6-Month Performance Trend</h4>
            <svg viewBox="0 0 400 150" className="w-full">
              {/* Grid lines */}
              <line x1="40" y1="20" x2="40" y2="120" stroke="#e5e7eb" strokeWidth="1" />
              <line x1="40" y1="120" x2="380" y2="120" stroke="#e5e7eb" strokeWidth="1" />
              
              {/* Trend lines */}
              <polyline points="40,100 100,85 160,75 220,65 280,55 340,40 380,30" fill="none" stroke="#3b82f6" strokeWidth="3" />
              <polyline points="40,110 100,105 160,100 220,95 280,88 340,80 380,75" fill="none" stroke="#ec4899" strokeWidth="2" strokeDasharray="5" />
              
              {/* Data points */}
              <circle cx="380" cy="30" r="4" fill="#3b82f6" />
              <circle cx="380" cy="75" r="4" fill="#ec4899" />
              
              {/* Labels */}
              <text x="40" y="140" fontSize="10" fill="#6b7280">Dec</text>
              <text x="160" y="140" fontSize="10" fill="#6b7280">Feb</text>
              <text x="280" y="140" fontSize="10" fill="#6b7280">Apr</text>
              <text x="380" y="140" fontSize="10" fill="#6b7280">May</text>
            </svg>
            
            <div className="flex justify-center gap-6 mt-4">
              <div className="flex items-center gap-2">
                <div className="w-4 h-1 bg-blue-500 rounded"></div>
                <span className="text-xs text-gray-600">{primaryBrand}</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-1 bg-pink-500 rounded"></div>
                <span className="text-xs text-gray-600">{competitor}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
