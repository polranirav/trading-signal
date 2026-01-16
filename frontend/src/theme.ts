import { createTheme, responsiveFontSizes, ThemeOptions } from '@mui/material/styles';

/**
 * Modern Tech Theme
 * Aesthetic: Futuristic, Fintech, High-Frequency Trading
 * Palette: Deep Navy, Electric Indigo, Neon Cyan
 */

// Core Palette
const colors = {
  background: {
    default: '#020617', // Deepest Slate/Navy
    paper: '#0f172a',   // Lighter Slate
  },
  primary: {
    main: '#6366f1', // Indigo 500
    light: '#818cf8',
    dark: '#4f46e5',
    contrastText: '#ffffff',
  },
  secondary: {
    main: '#8b5cf6', // Violet 500
    light: '#a78bfa',
    dark: '#7c3aed',
    contrastText: '#ffffff',
  },
  accent: {
    cyan: '#06b6d4',
    emerald: '#10b981',
    rose: '#f43f5e',
  },
  text: {
    primary: '#f8fafc', // Slate 50
    secondary: '#cbd5e1', // Slate 300
  }
};

const themeOptions: ThemeOptions = {
  palette: {
    mode: 'dark',
    primary: colors.primary,
    secondary: colors.secondary,
    background: colors.background,
    text: colors.text,
    success: {
      main: colors.accent.emerald,
    },
    error: {
      main: colors.accent.rose,
    },
    info: {
      main: colors.accent.cyan,
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 800,
      letterSpacing: '-0.025em',
      lineHeight: 1.1,
    },
    h2: {
      fontWeight: 700,
      letterSpacing: '-0.025em',
    },
    h3: {
      fontWeight: 700,
      letterSpacing: '-0.0125em',
    },
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
    button: {
      fontWeight: 600,
      textTransform: 'none',
      letterSpacing: '0.025em',
    },
  },
  shape: {
    borderRadius: 12, // Modern, slightly rounded but professional
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: colors.background.default,
          scrollbarColor: '#334155 #0f172a',
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: '#0f172a',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: '#334155',
            borderRadius: '4px',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          padding: '10px 24px',
          transition: 'all 0.2s ease-in-out',
        },
        contained: {
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          '&:hover': {
            transform: 'translateY(-1px)',
            boxShadow: '0 10px 15px -3px rgba(99, 102, 241, 0.3), 0 4px 6px -2px rgba(99, 102, 241, 0.1)',
          },
        },
        outlined: {
          borderWidth: '1px',
          '&:hover': {
            borderWidth: '1px',
            backgroundColor: 'rgba(99, 102, 241, 0.08)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none', // Disable default material dark mode overlay
        },
        rounded: {
          borderRadius: '16px',
        },
        elevation1: {
          backgroundColor: 'rgba(30, 41, 59, 0.4)', // Glassmorphism base
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        },
        elevation24: {
          backgroundColor: 'rgba(15, 23, 42, 0.7)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(99, 102, 241, 0.1)',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
        }
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(30, 41, 59, 0.3)',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          backgroundImage: 'none',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: '6px',
          fontWeight: 600,
        },
        filledPrimary: {
          background: 'rgba(99, 102, 241, 0.1)',
          color: '#818cf8',
          border: '1px solid rgba(99, 102, 241, 0.2)',
        }
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: 'rgba(30, 41, 59, 0.3)',
            backdropFilter: 'blur(4px)',
            '& fieldset': {
              borderColor: 'rgba(148, 163, 184, 0.2)',
            },
            '&:hover fieldset': {
              borderColor: '#6366f1',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#6366f1',
              borderWidth: '1px',
              boxShadow: '0 0 0 1px rgba(99, 102, 241, 0.2)',
            },
          },
        },
      },
    },
  },
};

const theme = createTheme(themeOptions);

export default responsiveFontSizes(theme);
