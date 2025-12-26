import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Sparkles, Zap, Target, CheckCircle2, ArrowRight, ArrowLeft, Crown, TrendingUp, FileText, ChevronRight, Brain, Search, RefreshCw, Shapes, Gauge, Wrench } from 'lucide-react';

export default function PrelaunchPage() {
    const [formData, setFormData] = useState({ name: '', email: '', company: '' });
    const [isSubmitted, setIsSubmitted] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [selectedChallengeIndex, setSelectedChallengeIndex] = useState(0);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);

        try {
            // Send data to Python backend
            const response = await fetch('http://localhost:5000/api/waitlist', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: formData.name,
                    email: formData.email,
                    company: formData.company
                })
            });

            const data = await response.json();

            if (data.success) {
                setIsLoading(false);
                setIsSubmitted(true);
                console.log('✅ Signup successful! Total signups:', data.total_signups);
            } else {
                setIsLoading(false);
                alert(data.error || 'Something went wrong. Please try again.');
            }
        } catch (error) {
            setIsLoading(false);
            console.error('Error submitting form:', error);
            alert('Failed to submit. Please check if the backend server is running.');
        }
    };

    const brandChallenges = [
        {
            icon: Brain,
            title: "STRATEGISE",
            whatYouFeel: "Every new campaign feels like a reset. Nobody remembers the last big narrative shift.",
            theRealCost: "You think you are building a compound story. Your market keeps receiving fresh noise.",
            componentsMissing: [
                "A memory that retains strategy, tone and rules with zero fatigue",
                "A brain that tracks competitor moves and market direction",
                "A diagnostic layer that can simulate how your choices will play out"
            ]
        },
        {
            icon: Search,
            title: "STUDY",
            whatYouFeel: "Your team spends hours researching competitors, but insights never translate into action.",
            theRealCost: "Analysis paralysis becomes your default state. Opportunities slip by while you're still gathering data.",
            componentsMissing: [
                "Real-time competitive intelligence that updates automatically",
                "Pattern recognition across market movements and messaging shifts",
                "Actionable insights that connect directly to your content strategy"
            ]
        },
        {
            icon: RefreshCw,
            title: "SYNC",
            whatYouFeel: "Your website says one thing, your sales deck says another, your emails sound like a third company.",
            theRealCost: "Prospects hear a different story at each touch. Consistency of trust never builds.",
            componentsMissing: [
                "A single source of truth for brand voice and positioning",
                "Automated consistency checks across all content channels",
                "Version control for messaging that keeps everyone aligned"
            ]
        },
        {
            icon: Shapes,
            title: "SHAPE",
            whatYouFeel: "You know your brand needs to evolve, but every change feels like starting from scratch.",
            theRealCost: "Your positioning stays frozen while the market moves. Relevance fades quarter by quarter.",
            componentsMissing: [
                "Adaptive positioning framework that evolves with market feedback",
                "A/B testing intelligence for messaging and narrative angles",
                "Gradual refinement system that prevents jarring brand pivots"
            ]
        },
        {
            icon: Gauge,
            title: "SENSE",
            whatYouFeel: "You publish content into the void. No idea what's working or why it resonates.",
            theRealCost: "You repeat mistakes and abandon winners. Every campaign is a coin flip.",
            componentsMissing: [
                "Performance tracking tied directly to brand narrative elements",
                "Sentiment analysis that reveals what truly resonates with your audience",
                "Predictive modeling for content performance before you hit publish"
            ]
        },
        {
            icon: Wrench,
            title: "SOLVE",
            whatYouFeel: "Creating content takes forever. Every piece requires starting from a blank page.",
            theRealCost: "Your team burns out on execution. Strategic thinking gets crowded out by production work.",
            componentsMissing: [
                "AI generation engine trained on your unique brand positioning",
                "Template library that maintains voice while accelerating output",
                "Quality assurance layer that ensures brand alignment at scale"
            ]
        }
    ];

    return (
        <div className="min-h-screen relative overflow-hidden" style={{ background: 'linear-gradient(135deg, #0F0C29 0%, #000000ff 50%, #24243E 100%)' }}>
            {/* Animated gradient orbs */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <motion.div
                    className="absolute -top-40 -left-40 w-96 h-96 rounded-full blur-[120px]"
                    style={{ background: 'radial-gradient(circle, rgba(255, 215, 0, 0.4), transparent 70%)' }}
                    animate={{
                        scale: [1, 1.2, 1],
                        opacity: [0.4, 0.6, 0.4],
                    }}
                    transition={{
                        duration: 8,
                        repeat: Infinity,
                        ease: "easeInOut"
                    }}
                />
                <motion.div
                    className="absolute top-1/3 -right-40 w-96 h-96 rounded-full blur-[120px]"
                    style={{ background: 'radial-gradient(circle, rgba(255, 107, 107, 0.4), transparent 70%)' }}
                    animate={{
                        scale: [1.2, 1, 1.2],
                        opacity: [0.5, 0.7, 0.5],
                    }}
                    transition={{
                        duration: 10,
                        repeat: Infinity,
                        ease: "easeInOut"
                    }}
                />
            </div>

            {/* Header */}
            <header className="relative z-50 border-b backdrop-blur-md" style={{ borderColor: 'rgba(255, 215, 0, 0.2)', backgroundColor: 'rgba(15, 12, 41, 0.8)' }}>
                <div className="max-w-7xl mx-auto px-8 py-4 flex items-center justify-between">
                    {/* Left: Back Arrow + Logo + 1SYX */}
                    <div className="flex items-center gap-4">
                        {/* Back Arrow Button */}
                        <button
                            onClick={() => {
                                window.location.href = '/';
                            }}
                            className="rounded-lg transition-all duration-300 hover:scale-110 flex items-center justify-center"
                            style={{
                                backgroundColor: 'rgba(255, 215, 0, 0.1)',
                                border: '1px solid rgba(255, 215, 0, 0.3)'
                            }}
                            aria-label="Back to Home"
                        >
                            <ArrowLeft className="w-5 h-5" style={{ color: '#FFD700' }} />
                        </button>

                        {/* Logo + Brand Name */}
                        <div className="flex items-center gap-3">
                            <img
                                src="/1syx-logo.jpeg"
                                alt="1SYX Logo"
                                className="h-10 w-auto"
                            />
                            <span className="text-2xl font-bold text-white tracking-wider">
                                1SYX
                            </span>
                        </div>
                    </div>

                    {/* Right: Join Waitlist Button */}
                    <button
                        onClick={() => {
                            document.getElementById('waitlist-section')?.scrollIntoView({
                                behavior: 'smooth',
                                block: 'center'
                            });
                        }}
                        className="px-6 py-3 font-bold rounded-xl transition-all duration-300 shadow-lg hover:scale-105 flex items-center gap-2"
                        style={{
                            background: 'linear-gradient(to right, #FFD700, #FF6B6B)',
                            color: '#000000',
                            boxShadow: '0 4px 15px rgba(255, 215, 0, 0.3)'
                        }}
                    >
                        <span>Join Waitlist</span>
                        <ArrowRight className="w-4 h-4" />
                    </button>
                </div>
            </header>

            <div className="relative py-20 px-8">
                {/* HERO SECTION - 50-50 split */}
                <div className="max-w-7xl mx-auto mb-24">
                    {/* Section Label */}
                    <div className="flex items-center gap-4 mb-8">
                        <span className="text-6xl font-bold" style={{ color: 'rgba(255, 215, 0, 0.15)' }}>01</span>
                        <div className="h-px flex-1" style={{ background: 'linear-gradient(to right, rgba(255, 215, 0, 0.3), transparent)' }}></div>
                    </div>

                    <div className="flex gap-8 items-start">
                        {/* LEFT: Hero Content - 50% width */}
                        <motion.div
                            initial={{ opacity: 0, x: -50 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.8 }}
                            style={{ width: '50%' }}
                        >
                            <h1 className="text-6xl xl:text-6xl font-bold mb-6 leading-tight" style={{ color: '#FFFFFF', marginBottom: '3rem' }}>
                                Your Brand's
                                <br />
                                <span
                                    className="text-transparent bg-clip-text"
                                    style={{
                                        backgroundImage: 'linear-gradient(to right, #FFD700, #FF6B6B, #FFD700)',
                                        WebkitBackgroundClip: 'text',
                                        WebkitTextFillColor: 'transparent'
                                    }}
                                >
                                    X-Factor
                                </span>
                                <br />
                                Engineered by AI
                            </h1>

                            <p className="text-xl leading-relaxed mb-8" style={{ color: '#B8B8D1', marginBottom: '3rem' }}>
                                The world's first <span className="font-semibold" style={{ color: '#FFFFFF' }}>Marketing Intelligence System</span> that
                                <br />
                                <span className="font-semibold" style={{ color: '#FFFFFF' }}>analyzes, strategizes, and generates</span> content
                                <br />
                                aligned with your competitive advantage.
                            </p>

                            {/* Value Props */}
                            <div className="space-y-4">
                                {[
                                    "No more generic content",
                                    "No more guessing your narrative",
                                    "No more copying competitors",
                                    "One system that aligns everything"
                                ].map((item, index) => (
                                    <motion.div
                                        key={item}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ duration: 0.5, delay: 0.4 + index * 0.1 }}
                                        className="flex items-center gap-3"
                                    >
                                        <div
                                            className="w-2 h-2 rounded-full"
                                            style={{ background: 'linear-gradient(to right, #FFD700, #FF6B6B)' }}
                                        />
                                        <span className="text-lg" style={{ color: '#D1D1E0' }}>{item}</span>
                                    </motion.div>
                                ))}
                            </div>
                        </motion.div>

                        {/* RIGHT: SNEAK PEAK Video - 50% width */}
                        <motion.div
                            initial={{ opacity: 0, x: 50 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.8, delay: 0.3 }}
                            style={{ width: '50%' }}
                        >
                            <div className="w-full">
                                {/* Video Container */}
                                <div
                                    className="backdrop-blur-xl border rounded-2xl p-6 shadow-2xl overflow-hidden"
                                    style={{
                                        backgroundColor: 'rgba(255, 255, 255, 0.05)',
                                        borderColor: 'rgba(255, 215, 0, 0.2)'
                                    }}
                                >
                                    <div className="relative w-full rounded-xl overflow-hidden" style={{ paddingBottom: '56.25%' }}>
                                        {/* Placeholder with animated gradient - replace with your video */}
                                        <div
                                            className="absolute top-0 left-0 w-full h-full flex items-center justify-center"
                                            style={{
                                                background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)'
                                            }}
                                        >
                                            <div className="text-center">
                                                <Sparkles className="w-16 h-16 mx-auto mb-4" style={{ color: '#FFD700' }} />
                                                <p className="text-xl font-semibold" style={{ color: '#FFFFFF' }}>Video Coming Soon</p>
                                                <p className="text-sm mt-2" style={{ color: '#B8B8D1' }}>Replace this placeholder with your video file</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                </div>

                {/* Section Divider */}
                <div className="max-w-7xl mx-auto" style={{ marginTop: '10rem', marginBottom: '10rem' }}>
                    <div className="h-px" style={{ background: 'linear-gradient(to right, transparent, rgba(255, 215, 0, 0.3), transparent)' }}></div>
                </div>

                {/* THE STACKED DAMAGE SECTION */}
                <div className="max-w-7xl mx-auto" style={{ marginBottom: '3rem' }}>
                    {/* Section Label */}
                    <div className="flex items-center gap-4 mb-8">
                        <span className="text-6xl font-bold" style={{ color: 'rgba(255, 107, 107, 0.15)' }}>02</span>
                        <div className="h-px flex-1" style={{ background: 'linear-gradient(to right, rgba(255, 107, 107, 0.3), transparent)' }}></div>
                    </div>

                    <div className="flex gap-12">
                        {/* LEFT: Header + Minimal Clickable Challenge List - 35% width */}
                        <motion.div
                            initial={{ opacity: 0, x: -30 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.8, delay: 0.7 }}
                            style={{ width: '35%' }}
                        >
                            {/* Section Header - Inside Left Column */}
                            <div className="mb-12" style={{ marginBottom: '2rem' }}>
                                <h2 className="text-3xl font-bold mb-4" style={{ color: '#FFFFFF', marginBottom: '2rem' }}>THE STACKED DAMAGE</h2>
                                <p className="text-lg leading-relaxed" style={{ color: '#8888A8' }}>
                                    Your brand is living with fourteen missing functions. Each one looks harmless on its own. Together they create a stack of consequences that bleeds attention, trust and revenue.
                                </p>
                            </div>

                            {/* Navigation List */}
                            <div className="space-y-1">
                                {brandChallenges.map((challenge, index) => (
                                    <motion.button
                                        key={challenge.title}
                                        onClick={() => setSelectedChallengeIndex(index)}
                                        className="w-full text-left px-4 py-3 transition-all duration-200 group flex items-center justify-between"
                                        style={{
                                            borderLeft: selectedChallengeIndex === index ? '3px solid #FFD700' : '3px solid transparent',
                                            backgroundColor: selectedChallengeIndex === index ? 'rgba(255, 215, 0, 0.05)' : 'transparent',
                                        }}
                                        whileHover={{ x: 4 }}
                                    >
                                        <div className="flex items-center gap-3">
                                            <challenge.icon
                                                className="w-4 h-4"
                                                style={{ color: selectedChallengeIndex === index ? '#FFD700' : '#6B6B8B' }}
                                            />
                                            <span
                                                className="font-medium text-base uppercase tracking-wide"
                                                style={{ color: selectedChallengeIndex === index ? '#FFFFFF' : '#8888A8' }}
                                            >
                                                {challenge.title}
                                            </span>
                                        </div>
                                        {selectedChallengeIndex === index && (
                                            <ChevronRight className="w-4 h-4" style={{ color: '#FFD700' }} />
                                        )}
                                    </motion.button>
                                ))}
                            </div>
                        </motion.div>

                        {/* RIGHT: Multi-Section Details Panel - 65% width */}
                        <motion.div
                            className="flex-1"
                            initial={{ opacity: 0, x: 30 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.8, delay: 0.8 }}
                        >
                            <AnimatePresence mode="wait">
                                <motion.div
                                    key={selectedChallengeIndex}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                    transition={{ duration: 0.3 }}
                                    className="space-y-6"
                                >
                                    {/* Section 1: This Is What You Feel */}
                                    <div
                                        className="backdrop-blur-sm border rounded-md p-6"
                                        style={{
                                            borderColor: 'rgba(210, 209, 206, 0.85)',
                                            backgroundColor: 'rgba(53, 51, 51, 0.05)'
                                        }}
                                    >
                                        <div className="flex items-center gap-2 mb-3">
                                            <ChevronRight className="w-4 h-4" style={{ color: '#FFD700' }} />
                                            <h3 className="text-lg font-semibold uppercase tracking-wider" style={{ color: '#8888A8' }}>
                                                THIS IS WHAT YOU FEEL
                                            </h3>
                                        </div>
                                        <p className="text-md leading-relaxed" style={{ color: '#FFFFFF' }}>
                                            "{brandChallenges[selectedChallengeIndex].whatYouFeel}"
                                        </p>
                                    </div>

                                    {/* Section 2: The Real Cost */}
                                    <div
                                        className="backdrop-blur-sm border rounded-md p-6"
                                        style={{
                                            backgroundColor: 'rgba(255, 107, 107, 0.05)',
                                            borderColor: 'rgba(255, 107, 107, 0.2)'
                                        }}
                                    >
                                        <div className="flex items-center gap-2 mb-3">
                                            <div
                                                className="w-2 h-2 rounded-full"
                                                style={{ backgroundColor: '#FF6B6B' }}
                                            />
                                            <h3 className="text-lg font-semibold uppercase tracking-wider" style={{ color: '#FF6B6B' }}>
                                                THE REAL COST
                                            </h3>
                                        </div>
                                        <p className="text-md leading-relaxed" style={{ color: '#D1D1E0' }}>
                                            {brandChallenges[selectedChallengeIndex].theRealCost}
                                        </p>
                                    </div>

                                    {/* Section 3: System Components Missing */}
                                    <div>
                                        <div className="flex items-center gap-2 mb-4">
                                            <div
                                                className="w-2 h-2 square-full"
                                                style={{ backgroundColor: '#eeeeeeff' }}
                                            />
                                            <h3 className="text-lg font-semibold uppercase tracking-wider" style={{ color: '#8888A8' }}>
                                                SYSTEM COMPONENTS MISSING
                                            </h3>
                                        </div>
                                        <div className="space-y-3">
                                            {brandChallenges[selectedChallengeIndex].componentsMissing.map((item, idx) => (
                                                <div
                                                    key={idx}
                                                    className="backdrop-blur-sm border rounded-md p-6"
                                                    style={{
                                                        backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                                        borderColor: 'rgba(255, 255, 255, 0.1)',
                                                        marginBottom: '1rem'
                                                    }}
                                                >
                                                    <div className="flex items-start gap-3">
                                                        <div
                                                            className="w-1.5 h-1.5 rounded-full mt-2 flex-shrink-0"
                                                            style={{ backgroundColor: '#8888A8' }}
                                                        />
                                                        <span className="text-md leading-relaxed" style={{ color: '#B8B8D1' }}>
                                                            {item}
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </motion.div>
                            </AnimatePresence>
                        </motion.div>
                    </div>
                </div>

                {/* Section Divider */}
                <div className="max-w-7xl mx-auto" style={{ marginTop: '10rem', marginBottom: '10rem' }}>
                    <div className="h-px" style={{ background: 'linear-gradient(to right, transparent, rgba(255, 215, 0, 0.3), transparent)' }}></div>
                </div>

                {/* WAITLIST FORM SECTION */}
                <div id="waitlist-section" className="max-w-5xl mx-auto" style={{ marginBottom: '6rem', width: '90%' }}>
                    {/* Section Label */}
                    <div className="flex items-center gap-4 mb-8">
                        <span className="text-6xl font-bold" style={{ color: 'rgba(255, 215, 0, 0.15)' }}>03</span>
                        <div className="h-px flex-1" style={{ background: 'linear-gradient(to right, rgba(255, 215, 0, 0.3), transparent)' }}></div>
                    </div>

                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.5 }}
                    >
                        <AnimatePresence mode="wait">
                            {!isSubmitted ? (
                                <motion.div
                                    key="form"
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.95 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <div
                                        className="backdrop-blur-xl border rounded-xl p-8 shadow-2xl"
                                        style={{
                                            backgroundColor: 'rgba(255, 255, 255, 0.08)',
                                            borderColor: 'rgba(255, 215, 0, 0.3)'
                                        }}
                                    >
                                        <div className="text-center mb-8">
                                            <h2 className="text-5xl font-bold mb-3" style={{ color: '#FFFFFF' }}>
                                                Join the Waitlist
                                            </h2>
                                            <p style={{ fontSize: '1.1rem', color: '#B8B8D1' }}>
                                                Be among the first to unlock your brand's X-Factor
                                            </p>
                                        </div>

                                        <form onSubmit={handleSubmit} className="space-y-5">
                                            <div>
                                                <label className="block text-md font-medium mb-2" style={{ color: '#D1D1E0' }}>
                                                    Full Name
                                                </label>
                                                <input
                                                    type="text"
                                                    placeholder="John Doe"
                                                    value={formData.name}
                                                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                                    required
                                                    className="w-full px-4 py-3 border rounded-xl focus:outline-none focus:ring-2 transition-all"
                                                    style={{
                                                        backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                                        borderColor: 'rgba(255, 215, 0, 0.3)',
                                                        color: '#FFFFFF',
                                                        marginBottom: '1rem'
                                                    }}
                                                />
                                            </div>

                                            <div>
                                                <label className="block text-md font-medium mb-2" style={{ color: '#D1D1E0' }}>
                                                    Work Email
                                                </label>
                                                <input
                                                    type="email"
                                                    placeholder="john@company.com"
                                                    value={formData.email}
                                                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                                    required
                                                    className="w-full px-4 py-3 border rounded-xl focus:outline-none focus:ring-2 transition-all"
                                                    style={{
                                                        backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                                        borderColor: 'rgba(255, 215, 0, 0.3)',
                                                        color: '#FFFFFF',
                                                        marginBottom: '1rem'
                                                    }}
                                                />
                                            </div>

                                            <div>
                                                <label className="block text-md font-medium mb-2" style={{ color: '#D1D1E0' }}>
                                                    Company Name
                                                </label>
                                                <input
                                                    type="text"
                                                    placeholder="Acme Inc."
                                                    value={formData.company}
                                                    onChange={(e) => setFormData({ ...formData, company: e.target.value })}
                                                    required
                                                    className="w-full px-4 py-3 border rounded-xl focus:outline-none focus:ring-2 transition-all"
                                                    style={{
                                                        backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                                        borderColor: 'rgba(255, 215, 0, 0.3)',
                                                        color: '#FFFFFF',
                                                        marginBottom: '1rem'
                                                    }}
                                                />
                                            </div>

                                            <button
                                                type="submit"
                                                disabled={isLoading}
                                                className="w-full mt-6 px-8 py-4 text-black font-bold rounded-xl transition-all duration-300 shadow-lg flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                                                style={{
                                                    background: 'linear-gradient(to right, #FFD700, #FF6B6B)',
                                                    boxShadow: '0 10px 30px rgba(255, 215, 0, 0.3)',
                                                    marginBottom: '1rem'
                                                }}
                                            >
                                                {isLoading ? (
                                                    <>
                                                        <motion.div
                                                            animate={{ rotate: 360 }}
                                                            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                                            className="w-5 h-5 border-2 border-black/30 border-t-black rounded-full"
                                                        />
                                                        <span>Joining...</span>
                                                    </>
                                                ) : (
                                                    <>
                                                        <span>Get Early Access</span>
                                                        <ArrowRight className="w-5 h-5" />
                                                    </>
                                                )}
                                            </button>

                                            <div className="flex items-center justify-center gap-6 text-xs pt-4" style={{ color: '#B8B8D1' }}>
                                                <div className="flex items-center gap-2">
                                                    <CheckCircle2 className="w-4 h-4" style={{ color: '#10B981' }} />
                                                    <span>No credit card</span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <CheckCircle2 className="w-4 h-4" style={{ color: '#10B981' }} />
                                                    <span>Cancel anytime</span>
                                                </div>
                                            </div>
                                        </form>

                                        <div className="mt-6 pt-6 border-t text-center" style={{ borderColor: 'rgba(255, 215, 0, 0.2)' }}>
                                            <p className="text-sm" style={{ color: '#B8B8D1' }}>
                                                Join <span className="font-semibold" style={{ color: '#FFFFFF' }}>10,000+</span> brands in the queue
                                            </p>
                                        </div>
                                    </div>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="success"
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.95 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <div
                                        className="backdrop-blur-xl border rounded-3xl p-10 shadow-2xl text-center"
                                        style={{
                                            backgroundColor: 'rgba(255, 255, 255, 0.08)',
                                            borderColor: 'rgba(255, 215, 0, 0.3)'
                                        }}
                                    >
                                        <motion.div
                                            initial={{ scale: 0 }}
                                            animate={{ scale: 1 }}
                                            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                                            className="w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6"
                                            style={{ background: 'rgba(16, 185, 129, 0.2)' }}
                                        >
                                            <CheckCircle2 className="w-10 h-10" style={{ color: '#10B981' }} />
                                        </motion.div>

                                        <h3 className="text-3xl font-bold mb-4" style={{ color: '#FFFFFF', marginBottom: '1rem' }}>
                                            You're In! 🎉
                                        </h3>

                                        <p className="mb-2" style={{ color: '#D1D1E0', marginBottom: '1rem' }}>
                                            Welcome to the future, <span className="font-semibold" style={{ color: '#FFFFFF' }}>{formData.name}</span>!
                                        </p>

                                        <p className="text-sm" style={{ color: '#B8B8D1', marginBottom: '1rem' }}>
                                            We'll send exclusive updates to <span style={{ color: '#FFD700' }}>{formData.email}</span>
                                        </p>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                </div>
            </div>

            {/* Footer */}
            <div
                className="relative px-8 py-6 flex items-center justify-between border-t backdrop-blur-sm mt-20"
                style={{
                    backgroundColor: 'rgba(15, 12, 41, 0.8)',
                    borderColor: 'rgba(255, 215, 0, 0.2)'
                }}
            >
                <p className="text-sm" style={{ color: '#8888A8' }}>&copy; 2024 1SYX. All rights reserved.</p>
                <div className="flex items-center gap-6 text-sm" style={{ color: '#8888A8' }}>
                    <a href="#" className="hover:text-white transition-colors">Privacy</a>
                    <a href="#" className="hover:text-white transition-colors">Terms</a>
                </div>
            </div>
        </div>
    );
}