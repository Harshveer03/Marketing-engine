import { Button } from "../ui/button";
import {
  ArrowRight,
  CheckCircle,
  Zap,
  Target,
  Sparkles,
  BarChart,
  Users,
  Clock,
  Plus,
  Minus,
  HelpCircle,
  Shield,
  DollarSign,
  Database,
  TrendingUp,
  MessageSquare,
  Lightbulb,
  RefreshCw,
  FileText,
  ThumbsUp,
  ThumbsDown,
  MessageCircle,
  Briefcase,
  Rocket,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { ImageWithFallback } from "../figma/ImageWithFallback";
import { useState } from "react";

interface LandingPageProps {
  onGetStarted: () => void;
}

export function LandingPage({ onGetStarted }: LandingPageProps) {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(null);

  const [helpfulVotes, setHelpfulVotes] = useState<{
    [key: number]: "yes" | "no" | null;
  }>({});

  const testimonials = [
    {
      name: "Sarah Johnson",
      role: "CEO, TechStart Inc",
      content:
        "1SYX transformed our brand strategy in just days. The AI insights were spot-on and helped us refine our messaging perfectly.",
      image:
        "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150&h=150&fit=crop",
    },
    {
      name: "Michael Chen",
      role: "Marketing Director, Fusion Co",
      content:
        "The speed and quality of brand materials we got was incredible. This tool saved us thousands in agency fees.",
      image:
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop",
    },
    {
      name: "Emily Rodriguez",
      role: "Founder, Creative Studio",
      content:
        "As a designer, I was skeptical at first, but 1SYX exceeded my expectations. It's now an essential part of our workflow.",
      image:
        "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop",
    },
  ];

  const faqs = [
    {
      icon: HelpCircle,
      question: "How does 1SYX work?",
      answer:
        "1SYX uses advanced AI to analyze your brand inputs and generate customized recommendations, strategies, and materials tailored to your specific needs. Simply share your brand information, and our AI will provide comprehensive insights and assets.",
      category: "Getting Started",
    },
    {
      icon: Clock,
      question: "How long does it take to get results?",
      answer:
        "You'll receive your brand score and initial recommendations within minutes. Complete brand materials and assets are typically ready within 24 hours, depending on the complexity of your requirements.",
      category: "Getting Started",
    },
    {
      icon: Target,
      question: "What kind of results can I expect?",
      answer:
        "You'll receive a comprehensive brand score, strategic recommendations, and ready-to-use brand materials including logos, color palettes, typography guidelines, and marketing content tailored to your industry and audience.",
      category: "Features",
    },
    {
      icon: Users,
      question: "Can I collaborate with my team?",
      answer:
        "Yes! 1SYX supports team collaboration features, allowing multiple stakeholders to contribute and review brand development. You can invite team members, share feedback, and work together seamlessly.",
      category: "Features",
    },
    {
      icon: DollarSign,
      question: "What are the pricing options?",
      answer:
        "We offer flexible pricing plans to suit businesses of all sizes. Start with our free trial to explore the platform, then choose from monthly or annual subscriptions. Enterprise plans are available for larger organizations with custom needs.",
      category: "Pricing",
    },
    {
      icon: Shield,
      question: "Is my data secure?",
      answer:
        "Absolutely. We use industry-standard encryption and security measures to protect your brand information and intellectual property. Your data is stored securely and never shared with third parties without your explicit consent.",
      category: "Security",
    },
  ];

  return (
    <div className="min-h-screen bg-white relative overflow-hidden">
      {/* Grid pattern background */}
      <div
        className="absolute inset-0 opacity-[0.15]"
        style={{
          backgroundImage: `linear-gradient(to right, #d1d5db 1px, transparent 1px), linear-gradient(to bottom, #d1d5db 1px, transparent 1px)`,
          backgroundSize: "40px 40px",
        }}
      />

      <div className="relative z-10">
        {/* Navigation */}
        <nav className="bg-white/95 backdrop-blur-md border-b border-gray-200 shadow-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-8 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <img
                src="/1syx-logo.jpeg"
                alt="1SYX Logo"
                className="h-10 w-auto"
              />
              <span className="text-xl font-bold text-gray-900 tracking-wider">
                1SYX
              </span>
            </div>

            <div className="flex items-center gap-8">
              <a
                href="#features"
                className="text-gray-600 hover:text-black transition-colors"
              >
                Features
              </a>
              <a
                href="#how-it-works"
                className="text-gray-600 hover:text-black transition-colors"
              >
                How It Works
              </a>
              <a
                href="#pricing"
                className="text-gray-600 hover:text-black transition-colors"
              >
                Pricing
              </a>
              <a
                href="#faq"
                className="text-gray-600 hover:text-black transition-colors"
              >
                FAQ
              </a>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={onGetStarted}
                  variant="outline"
                  className="!border-2 !border-gray-800 !text-gray-900 hover:!bg-gray-100 !bg-white font-semibold"
                >
                  Login
                </Button>
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={onGetStarted}
                  className="!bg-black hover:!bg-gray-800 !text-white !border-0 shadow-md"
                >
                  Get Started
                </Button>
              </motion.div>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <section className="max-w-7xl mx-auto px-8 py-20">
          <div className="grid grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <h1 className="text-6xl mb-8 font-bold text-gray-900">
                Your Brand's X-Factor, Engineered by One System
              </h1>
              <p className="text-xl text-gray-600 mb-6">
                AI-powered diagnostics, insights, and content — aligned
                perfectly to your strategy.
              </p>

              {/* Value Bullets */}
              <div className="space-y-4 mb-8">
                {[
                  "No more generic content",
                  "No more guessing your narrative",
                  "No more copying competitors",
                  "One system that aligns everything end-to-end",
                ].map((bullet, index) => (
                  <motion.div
                    key={bullet}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + index * 0.1 }}
                    className="flex items-center gap-4"
                  >
                    <CheckCircle className="w-5 h-5 flex-shrink-0" style={{ color: "#16a34a" }} />
                    <span className="text-gray-700 font-medium">{bullet}</span>
                  </motion.div>
                ))}
              </div>

              <div className="flex gap-4">
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    onClick={onGetStarted}
                    className="!bg-black hover:!bg-gray-800 !text-white !border-0 shadow-xl rounded-xl px-8 py-6 text-lg font-semibold"
                  >
                    <span className="flex items-center gap-2">
                      Start Your Diagnostic
                      <ArrowRight className="w-5 h-5" />
                    </span>
                  </Button>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    variant="outline"
                    className="!border-2 !border-gray-800 !text-gray-900 hover:!bg-gray-100 !bg-white rounded-xl px-8 py-6 text-lg font-semibold"
                  >
                    Watch Demo
                  </Button>
                </motion.div>
              </div>

              <div className="flex items-center gap-8 mt-12">
                <div>
                  <div className="text-3xl font-bold text-gray-900">10K+</div>
                  <div className="text-gray-600">Brands Optimized</div>
                </div>
                <div className="w-px h-12 bg-gray-300"></div>
                <div>
                  <div className="text-3xl font-bold text-gray-900">98%</div>
                  <div className="text-gray-600">Satisfaction Rate</div>
                </div>
                <div className="w-px h-12 bg-gray-300"></div>
                <div>
                  <div className="text-3xl font-bold text-gray-900">24/7</div>
                  <div className="text-gray-600">AI Support</div>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="relative"
            >
              <div className="bg-white rounded-3xl shadow-2xl p-8 relative overflow-hidden">
                <div
                  className="absolute inset-0 opacity-10 bg-cover bg-center"
                  style={{
                    backgroundImage: `url('https://images.unsplash.com/photo-1670225597315-782633cfbd2a?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtaW5pbWFsJTIwZ2VvbWV0cmljJTIwcGF0dGVybnxlbnwxfHx8fDE3NjMwMDUxMjF8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
                  }}
                />
                <ImageWithFallback
                  src="https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&h=600&fit=crop"
                  alt="Brand Dashboard"
                  className="rounded-2xl shadow-lg w-full relative z-10"
                />
              </div>
              <div className="absolute -top-6 -right-6 w-24 h-24 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full blur-2xl opacity-50"></div>
              <div className="absolute -bottom-6 -left-6 w-32 h-32 bg-gradient-to-br from-blue-400 to-indigo-500 rounded-full blur-2xl opacity-50"></div>
            </motion.div>
          </div>
        </section>

        {/* Sub-Hero - What 1SYX Does */}
        <section className="relative py-20 overflow-hidden">
          {/* Gradient Background */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-400 to-indigo-500 opacity-50" />

          <div className="max-w-4xl mx-auto px-8 text-center relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <h2 className="text-4xl font-bold text-gray-900 mb-4">
                1 System for Your X-Factor
              </h2>
              <p className="text-xl text-gray-700 mb-6 leading-relaxed">
                1SYX is the world's first{" "}
                <span className="font-bold">Marketing Intelligence System</span>{" "}
                that analyzes your brand, finds your competitive advantage, and
                generates content aligned with your strategy.
              </p>
              <div className="bg-white border-2 border-gray-900 rounded-xl p-6 inline-block">
                <p className="text-lg text-gray-900 font-semibold">
                  This isn't another AI tool.
                  <br />
                  It's your{" "}
                  <span className="text-black font-bold">
                    AI CMO + Strategist + Content Engine
                  </span>
                  .
                </p>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Trusted By Section */}
        <section className="bg-gray-50 py-12 border-y border-gray-200">
          <div className="max-w-7xl mx-auto px-8">
            <p className="text-center text-gray-900 mb-8 font-medium text-2xl">
              Trusted by teams, founders, and brands around the world
            </p>
            <div className="flex justify-center items-center gap-12">
              <div className="text-2xl font-bold text-gray-500">ACME Corp</div>
              <div className="text-2xl font-bold text-gray-500">TechVision</div>
              <div className="text-2xl font-bold text-gray-500">Innovate</div>
              <div className="text-2xl font-bold text-gray-500">
                BrightFuture
              </div>
              <div className="text-2xl font-bold text-gray-500">NextGen</div>
            </div>
          </div>
        </section>

        {/* The Problem Section */}
        <section className="bg-gray-900 py-20 relative overflow-hidden">
          <div
            className="absolute inset-0 opacity-30 bg-cover bg-center"
            style={{
              backgroundImage: `url('https://images.unsplash.com/photo-1557804506-669a67965ba0?w=1920&h=1080&fit=crop')`,
            }}
          />
          <div className="absolute inset-0 bg-gradient-to-b from-gray-900/90 via-gray-900/80 to-gray-900/90" />

          <div className="max-w-5xl mx-auto px-8 relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-center mb-12"
            >
              <h2 className="text-5xl mb-8 text-white font-bold leading-tight">
                Brands don't fail because of weak products.
                <br />
                They fail because of weak messaging.
              </h2>


              <div className="flex justify-center mb-8">
                <div className="grid grid-cols-2 gap-6 text-left max-w-4xl">
                  {[
                    "Your narrative sounds generic",
                    "Your ICP can't understand your value",
                    "Competitors dominate the conversation",
                    "Website & social messaging don't match",
                    "Content is inconsistent and random",
                    "Agencies are expensive, AI tools are too generic",
                  ].map((problem, index) => (
                    <motion.div
                      key={problem}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: 0.1 * index }}
                      className="flex items-start gap-3 bg-white/5 backdrop-blur-sm p-4 rounded-lg border border-white/10"
                    >
                      <div className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></div>
                      <span className="text-gray-300 font-medium">{problem}</span>
                    </motion.div>
                  ))}
                </div>
              </div>

              <div className="bg-white/10 backdrop-blur-sm border-2 border-white/20 rounded-xl p-8 inline-block">
                <p className="text-2xl text-white font-bold">
                  You don't need more content.
                  <br />
                  You need <span className="text-green-400">strategic content</span>.
                  <br />
                  <span className="text-white">1SYX delivers that.</span>
                </p>
              </div>
            </motion.div>
          </div>
        </section>

        {/* What is 1SYX Section */}
        <section className="relative py-20 overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-yellow-400 to-orange-500 opacity-50" />
          <div className="relative z-10 max-w-7xl mx-auto px-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-center mb-16"
            >
              <h2 className="text-5xl mb-4 font-bold text-black drop-shadow-sm">
                One System. Endless Strategic Power.
              </h2>
              <p className="text-xl text-gray-900 font-medium mb-12">
                1SYX is a unified AI system that powers your entire marketing strategy
              </p>
            </motion.div>

            <div className="grid grid-cols-2 gap-12 items-center mb-12">
              <motion.div
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="space-y-4"
              >
                {[
                  "Extracts your brand + competitor data",
                  "Diagnoses your messaging",
                  "Identifies your true market position",
                  "Generates insights & strategy",
                  "Produces content across all channels",
                  "Learns from performance and improves",
                ].map((item, index) => (
                  <motion.div
                    key={item}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.1 * index }}
                    className="flex items-center gap-4 bg-white backdrop-blur-sm p-4 rounded-lg border border-white/20 shadow-sm"
                  >
                    <div className="w-8 h-8 bg-black rounded-lg flex items-center justify-center flex-shrink-0">
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    </div>
                    <span className="text-gray-900 font-medium text-lg">{item}</span>
                  </motion.div>
                ))}
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="bg-white rounded-2xl p-12 text-center shadow-xl"
              >
                <Sparkles className="w-16 h-16 text-orange-500 mx-auto mb-6" />
                <h3 className="text-3xl font-bold text-gray-900 mb-4">
                  Your entire marketing
                </h3>
                <p className="text-xl text-gray-600">
                  <span className="text-green-600 font-bold">aligned</span>,{" "}
                  <span className="text-blue-600 font-bold">intelligent</span>,{" "}
                  <span className="text-purple-600 font-bold">automated</span>
                </p>
              </motion.div>
            </div>
          </div>
        </section>

        {/* The 6 Engines Section */}
        <section id="features" className="relative py-20 border-y border-gray-200 overflow-hidden">
          {/* Background Image */}
          <div
            className="absolute inset-0 bg-cover bg-center"
            style={{
              backgroundImage: `url('https://images.unsplash.com/photo-1552664730-d307ca884978?w=1920&h=1080&fit=crop')`,
            }}
          />
          {/* Light Overlay */}
          <div className="absolute inset-0 bg-white/85" />

          <div className="relative z-10 max-w-6xl mx-auto px-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-center mb-16"
            >
              <h2 className="text-5xl mb-4 font-bold text-gray-900">
                The 6 Engines of 1SYX
              </h2>
              <p className="text-xl text-gray-600">
                A complete system working together to power your brand
              </p>
            </motion.div>

            <div className="grid grid-cols-3 gap-8">
              {/* Engine 1: Extraction */}
              <EngineCard
                index={0}
                number="1"
                icon={Database}
                title="Extraction Engine"
                subtitle="All your inputs, intelligently structured"

              >
                <p className="text-gray-600 text-lg leading-relaxed">
                  Seamlessly ingests your website, social profiles, and documents to build a comprehensive structured brand profile.
                </p>
              </EngineCard>

              {/* Engine 2: Diagnostic */}
              <EngineCard
                index={1}
                number="2"
                icon={Target}
                title="Diagnostic Engine"
                subtitle="Your brand evaluated like a consulting firm would"

              >
                <p className="text-gray-600 text-lg leading-relaxed">
                  Evaluates your brand health across 7 key metrics and 4 strategic filters to identify gaps and opportunities.
                </p>
              </EngineCard>

              {/* Engine 3: Insight */}
              <EngineCard
                index={2}
                number="3"
                icon={Lightbulb}
                title="Insight Engine"
                subtitle="Analysis → Strategy"

              >
                <p className="text-gray-600 text-lg leading-relaxed">
                  Transforms raw diagnostic data into actionable positioning, narrative direction, and strategic priorities.
                </p>
              </EngineCard>

              {/* Engine 4: Content Generation */}
              <EngineCard
                index={3}
                number="4"
                icon={FileText}
                title="Content Generation Engine"
                subtitle="Content that aligns with your strategy — every time"

              >
                <p className="text-gray-600 text-lg leading-relaxed">
                  Instantly generates on-brand content for social, web, and video that aligns perfectly with your defined strategy.
                </p>
              </EngineCard>

              {/* Engine 5: Engagement Fetcher */}
              <EngineCard
                index={4}
                number="5"
                icon={TrendingUp}
                title="Engagement Fetcher Engine"
                subtitle="Performance visibility from every channel"

              >
                <p className="text-gray-600 text-lg leading-relaxed">
                  Aggregates performance data from all your active channels to give you clear visibility into what's driving results.
                </p>
              </EngineCard>

              {/* Engine 6: Analysis & Feedback Loop */}
              <EngineCard
                index={5}
                number="6"
                icon={RefreshCw}
                title="Analysis & Feedback Loop"
                subtitle="Your brand gets smarter over time"

              >
                <p className="text-gray-600 text-lg leading-relaxed">
                  Continuously learns from performance data to refine your messaging and improve your brand's impact over time.
                </p>
              </EngineCard>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section
          id="how-it-works"
          className="bg-gray-900 py-20 relative overflow-hidden"
        >
          <div
            className="absolute inset-0 opacity-10 bg-cover bg-center"
            style={{
              backgroundImage: `url('https://images.unsplash.com/photo-1557804506-669a67965ba0?w=1920&h=1080&fit=crop')`,
            }}
          />
          <div className="absolute inset-0 bg-gradient-to-b from-gray-900/90 via-gray-900/80 to-gray-900/90" />

          <div className="max-w-7xl mx-auto px-8 relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-center mb-16"
            >
              <h2 className="text-5xl mb-4 text-white">How It Works</h2>
              <p className="text-xl text-gray-300">
                Get your brand ready in 4 simple steps
              </p>
            </motion.div>

            <div className="grid grid-cols-4 gap-8">
              {[
                { number: "01", title: "Upload Your Inputs", description: "Provide details about your brand, industry, and audience." },
                { number: "02", title: "Run Your Diagnostic", description: "Receive a comprehensive brand analysis and recommendations." },
                { number: "03", title: "Review Insights", description: "Get strategic insights and positioning recommendations." },
                { number: "04", title: "Generate Content Aligned With Strategy", description: "Create professional brand materials instantly." },
              ].map((step, index) => (
                <motion.div
                  key={step.number}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  className="relative flex"
                >
                  <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 border border-white/20 hover:bg-white/20 transition-all duration-300 flex flex-col h-full w-full">
                    <div className="text-5xl font-bold text-white/30 mb-4">
                      {step.number}
                    </div>
                    <h3 className="text-xl mb-3 text-white">{step.title}</h3>
                    <p className="text-gray-300 flex-grow">
                      {step.description}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.6 }}
              className="text-center mt-12"
            >
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={onGetStarted}
                  className="bg-white text-black hover:bg-gray-100 border-0 shadow-xl rounded-xl px-8 py-6 text-lg"
                >
                  Start Your Brand Journey
                  <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </motion.div>
            </motion.div>
          </div>
        </section>

        {/* Outcomes Section */}
        <section className="px-8 py-20">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl mb-4 font-bold text-gray-900">
              Real Outcomes, Real Impact
            </h2>
            <p className="text-xl text-gray-600">
              Measurable results that transform your brand
            </p>
          </motion.div>

          <div className="max-w-7xl mx-auto">

            <div className="grid grid-cols-2 gap-8 mb-12">
              {/* Benefits Grid */}
              <div className="space-y-6">
                {[
                  "Crystal-clear messaging",
                  "Higher engagement",
                  "Stronger positioning",
                  "Consistent brand voice",
                ].map((benefit, index) => (
                  <motion.div
                    key={benefit}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.1 * index }}
                    className="bg-white p-6 rounded-lg border-2 border-green-200"
                  >
                    <div className="flex items-center gap-4">
                      <CheckCircle className="w-6 h-6 text-green-600" />
                      <span className="text-gray-900 font-semibold text-lg">{benefit}</span>
                    </div>
                  </motion.div>
                ))}
              </div>
              <div className="space-y-6">
                {[
                  "ICP-specific resonance",
                  "Faster growth",
                  "Real-time insights",
                  "Zero agency dependency",
                ].map((benefit, index) => (
                  <motion.div
                    key={benefit}
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.1 * index }}
                    className="bg-white p-6 rounded-lg border-2 border-purple-200"
                  >
                    <div className="flex items-center gap-4">
                      <CheckCircle className="w-6 h-6 text-purple-600" />
                      <span className="text-gray-900 font-semibold text-lg">{benefit}</span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Stats - Before/After Comparison */}
            <div className="grid grid-cols-4 gap-4 py-6 mb-12">
              {[
                { stat: "3×", label: "clearer messaging", before: "Confusing", after: "Crystal Clear" },
                { stat: "70%", label: "faster content cycles", before: "Weeks", after: "Days" },
                { stat: "40%", label: "average engagement uplift", before: "Low", after: "High" },
                { stat: "5-10hrs", label: "saved per week", before: "Manual", after: "Automated" },
              ].map((item, index) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.1 * index }}
                  className="bg-white rounded-xl p-8 shadow-lg border-2 border-gray-200 text-center"
                >
                  <div className="text-4xl font-bold text-gray-900 mb-2">{item.stat}</div>
                  <div className="text-gray-700 font-semibold mb-4">{item.label}</div>
                  <div className="flex items-center justify-center gap-2 text-sm">
                    <span className="bg-red-100 text-red-700 px-2 py-1 rounded">{item.before}</span>
                    <ArrowRight className="w-4 h-4 text-gray-400" />
                    <span className="bg-green-100 text-green-700 px-2 py-1 rounded">{item.after}</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Who Is It For Section */}
        <section className="relative px-8 py-20 border-y border-gray-200 overflow-hidden">
          {/* Background Image */}
          <div
            className="absolute inset-0 bg-cover bg-center"
            style={{
              backgroundImage: `url('https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=1920&h=1080&fit=crop')`,
            }}
          />
          {/* Dark Overlay */}
          <div className="absolute inset-0" style={{ backgroundColor: '#252424cc' }} />

          <div className="relative z-10 max-w-7xl mx-auto px-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-center mb-16"
            >
              <h2 className="text-5xl mb-4 font-bold text-white">
                Who Is It For?
              </h2>
              <p className="text-xl text-white">
                Built for anyone who wants to communicate better
              </p>
            </motion.div>

            <div className="grid grid-cols-4 gap-8 mb-12 justify-center">
              {[
                { icon: Rocket, label: "SaaS Founders" },
                { icon: Briefcase, label: "Agencies" },
                { icon: Users, label: "Marketing Teams" },
                { icon: Target, label: "Consultants" },
                { icon: Sparkles, label: "Creators" },
                { icon: Users, label: "Personal Brands" },
                { icon: TrendingUp, label: "DTC Brands" },
                { icon: BarChart, label: "B2B & B2C Companies" },
              ].map((item, index) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.05 * index }}
                  whileHover={{ scale: 1.05 }}
                  className="bg-white p-8 rounded-xl shadow-md border-2 border-gray-200 text-center hover:border-gray-900 transition-all"
                >
                  <item.icon className="w-15 h-15 text-gray-900 mx-auto mb-3" />
                  <p className="text-gray-900 font-semibold">{item.label}</p>
                </motion.div>
              ))}
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-center mb-12 py-12"
            >
              <div className="bg-white border-2 border-gray-900 rounded-xl p-6 inline-block">
                <p className="text-xl text-gray-900 font-bold">
                  If you speak to an audience,{" "}
                  <span className="text-black">1SYX helps you speak better.</span>
                </p>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Testimonials Section */}
        <section id="testimonials" className="max-w-7xl mx-auto px-8 py-20">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl mb-4">What Our Clients Say</h2>
            <p className="text-xl text-gray-600">
              Join thousands of satisfied brand builders
            </p>
          </motion.div>

          <div className="grid grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={testimonial.name}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                className="bg-white rounded-2xl p-8 shadow-lg border border-gray-200"
              >
                <div className="flex items-center gap-4 mb-6">
                  <ImageWithFallback
                    src={testimonial.image}
                    alt={testimonial.name}
                    className="w-16 h-16 rounded-full object-cover"
                  />
                  <div>
                    <div className="font-semibold text-gray-800">
                      {testimonial.name}
                    </div>
                    <div className="text-sm text-gray-600">
                      {testimonial.role}
                    </div>
                  </div>
                </div>
                <p className="text-gray-600 italic">"{testimonial.content}"</p>
                <div className="flex gap-1 mt-4">
                  {[...Array(5)].map((_, i) => (
                    <span key={i} className="text-yellow-400">
                      ★
                    </span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Pricing Section */}
        <section id="pricing" className="relative py-20 overflow-hidden">
          {/* Gradient Background */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-400 to-indigo-500 opacity-50" />

          <div className="relative z-10 max-w-7xl mx-auto px-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-center mb-16"
            >
              <h2 className="text-5xl mb-4 font-bold text-black">
                Simple, Transparent Pricing
              </h2>
              <p className="text-xl text-black">
                Choose the plan that fits your needs
              </p>
            </motion.div>

            <div className="grid grid-cols-4 gap-8">
              {/* Free Tier */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
                className="bg-white rounded-2xl p-8 shadow-lg border-2 border-gray-200 hover:border-gray-300 transition-all flex flex-col h-full"
              >
                <h3 className="text-2xl font-bold text-gray-900 mb-2">Free</h3>
                <p className="text-gray-600 mb-4">For exploring</p>
                <div className="mb-6">
                  <span className="text-5xl font-bold text-gray-900">$0</span>
                  <span className="text-gray-600">/mo</span>
                </div>
                <ul className="space-y-4 mb-8 flex-grow">
                  {["Basic diagnostics", "Limited uploads", "Sample insights", "Community support"].map((feature) => (
                    <li key={feature} className="flex items-start gap-4">
                      <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">{feature}</span>
                    </li>
                  ))}
                </ul>
                <Button
                  onClick={onGetStarted}
                  variant="outline"
                  className="w-full !border-2 !border-gray-800 !text-gray-900 hover:!bg-gray-100"
                >
                  Get Started
                </Button>
              </motion.div>

              {/* Starter Tier */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
                className="bg-white rounded-2xl p-8 shadow-lg border-2 border-gray-200 hover:border-gray-300 transition-all flex flex-col h-full"
              >
                <h3 className="text-2xl font-bold text-gray-900 mb-2">Starter</h3>
                <p className="text-gray-600 mb-4">For individuals & solopreneurs</p>
                <div className="mb-6">
                  <span className="text-5xl font-bold text-gray-900">$19</span>
                  <span className="text-gray-600">/mo</span>
                </div>
                <ul className="space-y-4 mb-8">
                  {["Basic diagnostics", "Limited PDF input", "Standard content generation", "Email support"].map((feature) => (
                    <li key={feature} className="flex items-start gap-2">
                      <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">{feature}</span>
                    </li>
                  ))}
                </ul>
                <Button
                  onClick={onGetStarted}
                  variant="outline"
                  className="w-full !border-2 !border-gray-800 !text-gray-900 hover:!bg-gray-100 mt-auto"
                >
                  Choose Starter
                </Button>
              </motion.div>

              {/* Growth Tier - Recommended */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3 }}
                className="bg-white rounded-2xl p-8 shadow-2xl border-4 border-black relative flex flex-col h-full"
              >
                <h3 className="text-2xl font-bold text-gray-900 mb-2 flex items-center gap-2 flex-wrap">
                  Growth
                  <div className="font-semibold px-1 py-1 rounded-[1px]" style={{ fontSize: "15px", backgroundColor: "#000000ff", color: "#ffffff", borderRadius: "12px" }}>
                    (Recommended)
                  </div>
                </h3>
                <p className="text-gray-600 mb-4">For startups & teams</p>
                <div className="mb-6">
                  <span className="text-5xl font-bold text-gray-900">$49</span>
                  <span className="text-gray-600">/mo</span>
                </div>
                <ul className="space-y-4 mb-8 flex-grow">
                  {["Advanced diagnostics", "Insight engine", "Multi-channel content engine", "Competitor analysis", "Priority support"].map((feature) => (
                    <li key={feature} className="flex items-start gap-2">
                      <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">{feature}</span>
                    </li>
                  ))}
                </ul>
                <Button
                  onClick={onGetStarted}
                  className="w-full !bg-black hover:!bg-gray-800 !text-white"
                >
                  Choose Growth
                </Button>
              </motion.div>

              {/* Enterprise Tier */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4 }}
                className="bg-white rounded-2xl p-8 shadow-lg border-2 border-gray-200 hover:border-gray-300 transition-all flex flex-col !h-full"
              >
                <h3 className="text-2xl font-bold text-gray-900 mb-2">Enterprise</h3>
                <p className="text-gray-600 mb-4">For organizations & agencies</p>
                <div className="mb-6">
                  <span className="text-5xl font-bold text-gray-900">Custom</span>
                </div>
                <ul className="space-y-4 mb-8">
                  {["Unlimited engines", "Custom rulebooks", "Team collaboration", "Dedicated support", "Custom integrations"].map((feature) => (
                    <li key={feature} className="flex items-start gap-2">
                      <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">{feature}</span>
                    </li>
                  ))}
                </ul>
                <Button
                  onClick={onGetStarted}
                  variant="outline"
                  className="w-full !border-2 !border-gray-800 !text-gray-900 hover:!bg-gray-100 mt-auto"
                >
                  Contact Sales
                </Button>
              </motion.div>
            </div>
          </div>
        </section >

        {/* FAQ Section */}
        < section id="faq" className="bg-white/40 py-20 border-y border-gray-200 " >
          <div className="max-w-7xl mx-auto px-8 ">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-center mb-16"
            >
              <h2 className="text-5xl mb-4">Frequently Asked Questions</h2>
              <p className="text-xl text-gray-600">
                Everything you need to know about 1SYX
              </p>
            </motion.div>

            <div className="grid grid-cols-2 gap-8 items-start">
              {/* Left Column */}
              <div className="space-y-6">
                {faqs.filter((_, i) => i % 2 === 0).map((faq, index) => {
                  const originalIndex = index * 2;
                  return (
                    <motion.div
                      key={originalIndex}
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.4, delay: index * 0.1 }}
                      className="bg-white rounded-2xl shadow-lg border-2 border-gray-200 overflow-hidden hover:shadow-xl hover:border-gray-300 transition-all duration-300 flex flex-col"
                    >
                      <button
                        onClick={() =>
                          setOpenFaqIndex(openFaqIndex === originalIndex ? null : originalIndex)
                        }
                        className="w-full p-6 flex items-center justify-between text-left transition-colors duration-200"
                      >
                        <div className="flex items-center gap-4 flex-1">
                          <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center flex-shrink-0">
                            <span className="text-sm font-bold text-gray-700">
                              {originalIndex + 1}
                            </span>
                          </div>

                          <div className="w-12 h-12 bg-black rounded-xl flex items-center justify-center flex-shrink-0">
                            <faq.icon className="w-6 h-6 text-white" />
                          </div>

                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                {faq.category}
                              </span>
                            </div>
                            <h3 className="text-lg font-bold text-gray-900">
                              {faq.question}
                            </h3>
                          </div>
                        </div>
                        <motion.div
                          animate={{ rotate: openFaqIndex === originalIndex ? 180 : 0 }}
                          transition={{ duration: 0.3 }}
                          className="flex-shrink-0 ml-4"
                        >
                          {openFaqIndex === originalIndex ? (
                            <Minus className="w-6 h-6 text-gray-900" />
                          ) : (
                            <Plus className="w-6 h-6 text-gray-900" />
                          )}
                        </motion.div>
                      </button>

                      <AnimatePresence>
                        {openFaqIndex === originalIndex && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3 }}
                            className="overflow-hidden"
                          >
                            <div className="px-6 pb-6 pt-2 border-t border-gray-100">
                              <div className="pl-20 pr-10">
                                <p className="text-gray-700 leading-relaxed mb-6">
                                  {faq.answer}
                                </p>

                                <div className="flex items-center gap-4 pt-4 border-t border-gray-100">
                                  <span className="text-sm text-gray-600 font-medium">
                                    Was this helpful?
                                  </span>
                                  <div className="flex gap-2">
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setHelpfulVotes({
                                          ...helpfulVotes,
                                          [originalIndex]: "yes",
                                        });
                                      }}
                                      className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all duration-200 ${helpfulVotes[originalIndex] === "yes"
                                        ? "bg-green-50 border-green-500 text-green-700"
                                        : "border-gray-300 text-gray-600 hover:bg-gray-50"
                                        }`}
                                    >
                                      <ThumbsUp className="w-4 h-4" />
                                      <span className="text-sm font-medium">
                                        Yes
                                      </span>
                                    </button>
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setHelpfulVotes({
                                          ...helpfulVotes,
                                          [originalIndex]: "no",
                                        });
                                      }}
                                      className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all duration-200 ${helpfulVotes[originalIndex] === "no"
                                        ? "bg-red-50 border-red-500 text-red-700"
                                        : "border-gray-300 text-gray-600 hover:bg-gray-50"
                                        }`}
                                    >
                                      <ThumbsDown className="w-4 h-4" />
                                      <span className="text-sm font-medium">
                                        No
                                      </span>
                                    </button>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  );
                })}
              </div>

              {/* Right Column */}
              <div className="space-y-6">
                {faqs.filter((_, i) => i % 2 !== 0).map((faq, index) => {
                  const originalIndex = index * 2 + 1;
                  return (
                    <motion.div
                      key={originalIndex}
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.4, delay: index * 0.1 }}
                      className="bg-white rounded-2xl shadow-lg border-2 border-gray-200 overflow-hidden hover:shadow-xl hover:border-gray-300 transition-all duration-300 flex flex-col"
                    >
                      <button
                        onClick={() =>
                          setOpenFaqIndex(openFaqIndex === originalIndex ? null : originalIndex)
                        }
                        className="w-full p-6 flex items-center justify-between text-left hover:bg-gray-50 transition-colors duration-200"
                      >
                        <div className="flex items-center gap-4 flex-1">
                          <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center flex-shrink-0">
                            <span className="text-sm font-bold text-gray-700">
                              {originalIndex + 1}
                            </span>
                          </div>

                          <div className="w-12 h-12 bg-black rounded-xl flex items-center justify-center flex-shrink-0">
                            <faq.icon className="w-6 h-6 text-white" />
                          </div>

                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                {faq.category}
                              </span>
                            </div>
                            <h3 className="text-lg font-bold text-gray-900">
                              {faq.question}
                            </h3>
                          </div>
                        </div>
                        <motion.div
                          animate={{ rotate: openFaqIndex === originalIndex ? 180 : 0 }}
                          transition={{ duration: 0.3 }}
                          className="flex-shrink-0 ml-4"
                        >
                          {openFaqIndex === originalIndex ? (
                            <Minus className="w-6 h-6 text-gray-900" />
                          ) : (
                            <Plus className="w-6 h-6 text-gray-900" />
                          )}
                        </motion.div>
                      </button>

                      <AnimatePresence>
                        {openFaqIndex === originalIndex && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3 }}
                            className="overflow-hidden"
                          >
                            <div className="px-6 pb-6 pt-2 border-t border-gray-100">
                              <div className="pl-20 pr-10">
                                <p className="text-gray-700 leading-relaxed mb-6">
                                  {faq.answer}
                                </p>

                                <div className="flex items-center gap-4 pt-4 border-t border-gray-100">
                                  <span className="text-sm text-gray-600 font-medium">
                                    Was this helpful?
                                  </span>
                                  <div className="flex gap-2">
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setHelpfulVotes({
                                          ...helpfulVotes,
                                          [originalIndex]: "yes",
                                        });
                                      }}
                                      className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all duration-200 ${helpfulVotes[originalIndex] === "yes"
                                        ? "bg-green-50 border-green-500 text-green-700"
                                        : "border-gray-300 text-gray-600 hover:bg-gray-50"
                                        }`}
                                    >
                                      <ThumbsUp className="w-4 h-4" />
                                      <span className="text-sm font-medium">
                                        Yes
                                      </span>
                                    </button>
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setHelpfulVotes({
                                          ...helpfulVotes,
                                          [originalIndex]: "no",
                                        });
                                      }}
                                      className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all duration-200 ${helpfulVotes[originalIndex] === "no"
                                        ? "bg-red-50 border-red-500 text-red-700"
                                        : "border-gray-300 text-gray-600 hover:bg-gray-50"
                                        }`}
                                    >
                                      <ThumbsDown className="w-4 h-4" />
                                      <span className="text-sm font-medium">
                                        No
                                      </span>
                                    </button>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  );
                })}
              </div>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4 }}
              className="mt-12 text-center"
            >
              <div className="relative rounded-2xl p-8 border-2 border-gray-200 overflow-hidden">
                <div
                  className="absolute inset-0 opacity-70"
                  style={{
                    backgroundImage: `url('https://images.unsplash.com/photo-1521737604893-d14cc237f11d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2084&q=80')`,
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                  }}
                />
                <div
                  className="absolute inset-0 rounded-2xl"
                  style={{
                    background:
                      "linear-gradient(to bottom right, rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.85))",
                  }}
                />
                <div className="relative z-10">
                  <MessageCircle className="w-12 h-12 text-white mx-auto mb-4" />
                  <h3 className="text-2xl font-bold text-white mb-2">
                    Still have questions?
                  </h3>
                  <p className="text-white mb-6">
                    Can't find the answer you're looking for? Our support team is
                    here to help.
                  </p>
                  <div className="flex gap-4 justify-center">
                    <Button
                      onClick={onGetStarted}
                      className="!bg-black hover:!bg-gray-800 !text-white !border-0 shadow-lg"
                    >
                      Contact Support
                    </Button>
                    <Button
                      variant="outline"
                      className="!border-2 !border-gray-800 !text-gray-900 hover:!bg-gray-100 !bg-white"
                    >
                      View Documentation
                    </Button>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </section >

        {/* Final CTA Section */}
        <section className="py-20 relative">
          <div className="absolute inset-0 bg-gradient-to-br from-yellow-400 to-orange-500 opacity-50" />
          <div className="max-w-7xl mx-auto px-8 relative z-10">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="rounded-3xl p-16 text-center relative overflow-hidden shadow-2xl border-2 border-gray-300"
            >
              <div
                className="absolute inset-0 rounded-3xl"
                style={{
                  backgroundImage: `url('https://images.unsplash.com/photo-1542744173-8e7e53415bb0?w=1920&h=1080&fit=crop')`,
                  backgroundSize: "cover",
                  backgroundPosition: "center",
                }}
              />

              <div
                className="absolute inset-0 rounded-3xl"
                style={{
                  background:
                    "linear-gradient(to bottom right, rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.85))",
                }}
              />

              <div className="relative z-10">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.2 }}
                  className="inline-flex items-center gap-2 bg-white/20 backdrop-blur-sm border border-white/30 rounded-full px-6 py-2 mb-6"
                >
                  <Sparkles className="w-5 h-5" style={{ color: "#fde047" }} />
                  <span
                    className="font-semibold text-sm"
                    style={{ color: "#ffffff" }}
                  >
                    Transform Your Brand Today
                  </span>
                </motion.div>

                <h2
                  className="text-5xl md:text-6xl mb-6 font-bold leading-tight"
                  style={{ color: "#ffffff" }}
                >
                  Ready to Unlock Your X-Factor?
                </h2>
                <p
                  className="text-lg md:text-xl mb-10 max-w-2xl mx-auto leading-relaxed"
                  style={{ color: "#e5e7eb" }}
                >
                  Join thousands of brands using 1SYX to transform their messaging,
                  sharpen their narrative, and dominate their category.
                </p>

                <div className="flex gap-4 justify-center mb-6">
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Button
                      onClick={onGetStarted}
                      className="bg-white text-black hover:bg-gray-100 border-0 shadow-xl rounded-xl px-10 py-6 text-lg font-semibold"
                    >
                      Get Started Free
                      <ArrowRight className="ml-2 w-5 h-5" />
                    </Button>
                  </motion.div>
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Button
                      variant="outline"
                      className="!border-2 !border-white !text-white hover:!bg-white/10 !bg-transparent rounded-xl px-10 py-6 text-lg font-semibold"
                    >
                      Schedule a Demo
                    </Button>
                  </motion.div>
                </div>

                <div className="flex items-center justify-center gap-8 text-sm">
                  <div className="flex items-center gap-2">
                    <CheckCircle
                      className="w-5 h-5"
                      style={{ color: "#4ade80" }}
                    />
                    <span className="font-medium" style={{ color: "#e5e7eb" }}>
                      No credit card required
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle
                      className="w-5 h-5"
                      style={{ color: "#4ade80" }}
                    />
                    <span className="font-medium" style={{ color: "#e5e7eb" }}>
                      14-day free trial
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle
                      className="w-5 h-5"
                      style={{ color: "#4ade80" }}
                    />
                    <span className="font-medium" style={{ color: "#e5e7eb" }}>
                      Cancel anytime
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </section >

        {/* Footer */}
        < footer className="bg-gray-900 text-gray-300 py-12" >
          <div className="max-w-7xl mx-auto px-8">
            <div className="grid grid-cols-4 gap-8 mb-8">
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <img
                    src="/1syx-logo.jpeg"
                    alt="1SYX Logo"
                    className="h-8 w-auto"
                  />
                  <span className="text-xl font-semibold text-white">1SYX</span>
                </div>
                <p className="text-gray-400">1 System for Your X-Factor</p>
              </div>

              <div>
                <h4 className="text-white mb-4">Product</h4>
                <ul className="space-y-2">
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Features
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Pricing
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Updates
                    </a>
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="text-white mb-4">Company</h4>
                <ul className="space-y-2">
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      About
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Blog
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Careers
                    </a>
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="text-white mb-4">Legal</h4>
                <ul className="space-y-2">
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Privacy
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Terms
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Contact
                    </a>
                  </li>
                </ul>
              </div>
            </div>

            <div className="border-t border-gray-800 pt-8 flex justify-between items-center">
              <p className="text-gray-500">© 2024 1SYX. All rights reserved.</p>
              <div className="flex gap-4">
                <a
                  href="#"
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  Twitter
                </a>
                <a
                  href="#"
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  LinkedIn
                </a>
                <a
                  href="#"
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  Instagram
                </a>
              </div>
            </div>
          </div>
        </footer >
      </div >
    </div >
  );
}

// Engine Card Component
interface EngineCardProps {
  index: number;
  number: string;
  icon: React.ElementType;
  title: string;
  subtitle: string;
  children: React.ReactNode;
}

function EngineCard({
  index,
  number,
  icon: Icon,
  title,
  subtitle,
  children,
}: EngineCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
      className="bg-white rounded-2xl shadow-lg border-2 border-gray-200 overflow-hidden hover:shadow-xl hover:border-gray-300 transition-all duration-300 flex flex-col h-full p-8"
    >
      <div className="flex items-start justify-between mb-6">
        <div className="w-14 h-14 bg-black rounded-xl flex items-center justify-center flex-shrink-0">
          <Icon className="w-7 h-7 text-white" />
        </div>
        <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center flex-shrink-0">
          <span className="text-lg font-bold text-gray-700">{number}</span>
        </div>
      </div>

      <h3 className="text-xl font-bold text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600 font-medium mb-4">{subtitle}</p>

      <div className="mt-auto pt-4 border-t border-gray-100">
        {children}
      </div>
    </motion.div>
  );
}
