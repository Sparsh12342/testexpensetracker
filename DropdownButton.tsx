import React, { useState } from "react";

function DropdownButton() {
  const [isOpen, setIsOpen] = useState(false);
  const toggleDropdown = () => setIsOpen(!isOpen);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      console.log("Uploaded file:", file.name);
      // You can pass this file to a backend or read with FileReader
    }
  };

  return (
    <div className="relative inline-block text-left mb-4 space-y-4">
      {/* Dropdown Button */}
      <button
        onClick={toggleDropdown}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      >
        Detailed Summary
      </button>

      {isOpen && (
        <div className="absolute mt-2 w-48 bg-blue-500 border border-blue-700 rounded shadow-lg z-10">
          <ul>
            <li className="px-4 py-2 text-black hover:bg-blue-600 hover:text-white cursor-pointer">
              Bars and Pubs
            </li>
            <li className="px-4 py-2 text-black hover:bg-blue-600 hover:text-white cursor-pointer">
              Clothing & Apparel
            </li>
            <li className="px-4 py-2 text-black hover:bg-blue-600 hover:text-white cursor-pointer">
              Food & Dining
            </li>
          </ul>
        </div>
      )}

      {/* File Upload Button */}
      <label className="block w-fit px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 cursor-pointer">
        Upload CSV
        <input
          type="file"
          accept=".csv"
          onChange={handleFileUpload}
          className="hidden"
        />
      </label>
    </div>
  );
}

export default DropdownButton;
