import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "D-FINE Tiny-Target Accuracy Lab",
  description:
    "Test D-FINE human and vehicle detection on images, video, mobile cameras and webcams with an optional precision tile scan.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
