import { Route, Routes } from "react-router-dom";
import { Header } from "./components/Header";
import { Dashboard } from "./pages/Dashboard";
import { Landing } from "./pages/Landing";
import { SignIn } from "./pages/SignIn";
import { Workspace } from "./pages/Workspace";

export default function App() {
  return (
    <>
      {/* First tab stop on every page. The landing hero is tall, and without
       * this a keyboard user tabs through the entire nav before reaching
       * content. */}
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <Header />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/signin" element={<SignIn />} />
        {/* Redirecting here is UX, not security. The server enforces
         * ownership on every profile-scoped endpoint; a hidden route is not
         * an access control and is not treated as one. */}
        <Route path="/app" element={<Workspace />} />
        <Route path="/ops" element={<Dashboard />} />
        <Route path="*" element={<Landing />} />
      </Routes>
    </>
  );
}
