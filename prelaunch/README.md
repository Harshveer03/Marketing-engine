# BrandGen - AI-Powered Brand Discovery Platform

A modern, full-featured brand discovery and strategy platform built with React, TypeScript, and Tailwind CSS. BrandGen helps businesses create, refine, and scale their brand identity through an intuitive multi-step form process with AI-powered insights.

## ✨ Features

### 🎨 Landing Page

- Beautiful, responsive landing page with smooth animations
- Feature showcase with 6 key capabilities
- Step-by-step "How It Works" section
- Customer testimonials
- FAQ section
- Full marketing funnel design

### 🔐 Authentication System

- Login and Sign-up functionality
- Work email validation (blocks personal email domains)
- Password validation (minimum 8 characters, 1 uppercase letter)
- Persistent login with localStorage
- User profile menu with logout
- Secure session management

### 📋 Multi-Step Form

Three main sections for brand discovery:

1. **What Use?** - Define your brand goals (Refine, Redefine, From Scratch, General Use)
2. **Brand Info & Resources** - Collect brand information, upload assets, and add reference links
3. **Score** - Review completion status and get brand score (85/100)

### 🎯 UI/UX Features

- Smooth page transitions with Framer Motion
- Gradient backgrounds and modern design
- Responsive layout
- Interactive hover effects
- Progress tracking
- Real-time form validation

## 🚀 Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS v4** - Styling
- **Framer Motion** - Animations
- **Radix UI** - Accessible component primitives
- **shadcn/ui** - Pre-built components
- **Lucide React** - Icons

## 📦 Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd Landing-Page-Form-Sections
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

4. Open your browser and navigate to:

```
http://localhost:3000
```


## 🏗️ Project Structure

```
├── src/
│   ├── components/
│   │   ├── ui/              # shadcn/ui components
│   │   ├── figma/           # Figma-specific components
│   │   ├── AuthPage.tsx     # Login/Sign-up page
│   │   ├── LandingPage.tsx  # Marketing landing page
│   │   ├── Header.tsx       # Top navigation header
│   │   ├── TopNav.tsx       # Secondary navigation
│   │   ├── Sidebar.tsx      # Section navigation
│   │   ├── WhatUseSection.tsx
│   │   ├── BrandInfoSection.tsx
│   │   └── ScoreSection.tsx
│   ├── styles/
│   │   └── globals.css      # Global styles
│   ├── App.tsx              # Main app component
│   ├── main.tsx             # Entry point
│   └── index.css            # Tailwind imports
├── public/
├── .gitignore
├── package.json
├── vite.config.ts
└── README.md
```

## 🎨 Design System

- **Primary Colors:** Indigo (500-700) and Purple (500-700)
- **Gradients:** Blue-50 → Indigo-50 → Purple-50
- **Typography:** System fonts with custom sizing
- **Border Radius:** Rounded-xl (1rem) and Rounded-2xl (1.5rem)
- **Shadows:** Multi-layer shadows for depth

## 🔧 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## 🌟 Key Features Explained

### Authentication Flow

1. User clicks "Get Started" on landing page
2. Redirected to authentication page
3. Can toggle between Login and Sign-up
4. Work email validation ensures business emails only
5. Session persists across page refreshes
6. User can logout from profile menu

### Form Flow

1. **What Use?** - Select your use case
2. **Brand Info** - Enter brand details and upload resources
3. **Score** - View completion status and brand score
4. Navigation via sidebar or "Next" buttons

### Validation Rules

- **Email:** Must be valid format and work domain
- **Password:** Minimum 8 characters, at least 1 uppercase letter
- **Work Email Domains Blocked:** gmail.com, yahoo.com, hotmail.com, outlook.com, etc.

## 📱 Responsive Design

The application is fully responsive and works on:

- Desktop (1920px+)
- Laptop (1024px - 1919px)
- Tablet (768px - 1023px)
- Mobile (320px - 767px)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project uses components from [shadcn/ui](https://ui.shadcn.com/) under the [MIT License](https://github.com/shadcn-ui/ui/blob/main/LICENSE.md).

Images from [Unsplash](https://unsplash.com) used under their [license](https://unsplash.com/license).

## 🙏 Acknowledgments

- Original Figma design: [Landing Page Form Sections](https://www.figma.com/design/0qqLYHZ1OjrLytnBY3ZclZ/Landing-Page-Form-Sections)
- shadcn/ui for the component library
- Radix UI for accessible primitives
- Unsplash for images

## 📞 Support

For support, please open an issue in the GitHub repository.

---

Built with ❤️ using React, TypeScript, and Tailwind CSS
