import React, { useState } from "react";
import DropdownButton from "./DropdownButton";
import Papa from "papaparse";

import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
} from "recharts";

// Mirrors backend response structure
interface FirstWordSummary {
  FirstWord: string;
  TotalDeposits?: number;
  TotalWithdrawals?: number;
}
interface DepositCluster {
  Cluster_Label: string;
  TotalDeposits: number;
}
interface WithdrawalCluster {
  Cluster_Label: string;
  TotalWithdrawals: number;
}
interface CategorySummary {
  Category: string;
  TransactionCount: number;
  TotalAmount: number;
  Withdrawals: number;
  Deposits: number;
}
interface SummaryData {
  total_deposits: number;
  total_withdrawals: number;
  num_deposits: number;
  num_withdrawals: number;
  deposits_grouped_by_first_word: FirstWordSummary[];
  withdrawals_grouped_by_first_word: FirstWordSummary[];
  deposits_grouped_by_cluster: DepositCluster[];
  withdrawals_grouped_by_cluster: WithdrawalCluster[];
  category_summary: CategorySummary[];
}

const categoryOptions = [
  "Bars & Pubs",
  "Clothing & Apparel",
  "Food & Dining",
  "Retail",
  "Transfers",
];
const pieColors: { [key: string]: string } = {
  "Bars & Pubs": "#ff9999",
  "Clothing & Apparel": "#66b3ff",
  "Food & Dining": "#99ff99",
  Retail: "#ffcc99",
  Transfers: "#c299ff",
};

function App() {
  const [data, setData] = useState<SummaryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [entries, setEntries] = useState<string[][]>([]);
  const [entry, setEntry] = useState({
    Date: "",
    Description: "",
    Amount: "",
    Category: categoryOptions[0],
  });

  const format = (n: number) =>
    n >= 0 ? `$${n.toFixed(2)}` : `-$${Math.abs(n).toFixed(2)}`;
  const getCategoryColor = (cat: string) =>
    ({
      "Bars & Pubs": "#ffe5e5",
      "Clothing & Apparel": "#e5f0ff",
      "Food & Dining": "#e5ffe5",
      Retail: "#fff5cc",
      Transfers: "#f0e5ff",
    }[cat] || "#f9f9f9");
  const getPieColor = (cat: string) => pieColors[cat] || "#dddddd";

  // Send combined manual + CSV entries to backend
  const processData = async (file: File | null, manual: string[][]) => {
    if (!file && manual.length === 0) return;
    setLoading(true);
    const headers = ["Date", "Description", "Amount", "Category"];
    const allData = [headers, ...manual];
    if (file) {
      const text = await file.text();
      const parsed = Papa.parse<string[]>(text, { skipEmptyLines: true });
      allData.push(...parsed.data.slice(1));
    }
    const blob = new Blob([Papa.unparse(allData)], { type: "text/csv" });
    const form = new FormData();
    form.append("file", blob, "combined.csv");
    try {
      const res = await fetch("http://localhost:5050/upload-csv", {
        method: "POST",
        body: form,
      });
      const json: SummaryData = await res.json();
      setData(json);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  // Sort descending by date string
  const sortByDateDesc = (rows: string[][]) =>
    [...rows].sort(
      (a, b) => new Date(b[0]).getTime() - new Date(a[0]).getTime()
    );

  // CSV upload: parse, merge, update entries & summary
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    const text = await file.text();
    const parsed = Papa.parse<string[]>(text, { skipEmptyLines: true });
    const csvRows = parsed.data.slice(1);
    const combined = sortByDateDesc([...entries, ...csvRows]);
    setEntries(combined);
    await processData(null, combined);
    setLoading(false);
  };

  const handleEntryChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setEntry((prev) => ({ ...prev, [name]: value }));
  };

  const addManualEntry = () => {
    if (!entry.Date || !entry.Description || !entry.Amount) return;
    const newRow = [
      entry.Date,
      entry.Description,
      entry.Amount,
      entry.Category,
    ];
    const combined = sortByDateDesc([...entries, newRow]);
    setEntries(combined);
    setEntry({
      Date: "",
      Description: "",
      Amount: "",
      Category: categoryOptions[0],
    });
    processData(null, combined);
  };

  return (
    <div
      style={{
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        padding: 20,
        maxWidth: 900,
        margin: "0 auto",
      }}
    >
      <DropdownButton />
      <h1 style={{ textAlign: "center", marginBottom: 20 }}>
        Spending Summary
      </h1>

      {/* Manual Entry & CSV Upload */}
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          marginBottom: 10,
          tableLayout: "fixed",
        }}
      >
        <thead>
          <tr style={{ backgroundColor: "#333", color: "white" }}>
            <th style={{ padding: 8 }}>Date</th>
            <th style={{ padding: 8 }}>Description</th>
            <th style={{ padding: 8 }}>Amount</th>
            <th style={{ padding: 8 }}>Category</th>
            <th style={{ padding: 8 }}>Add</th>
          </tr>
        </thead>
        <tbody>
          <tr style={{ backgroundColor: "#f7f7f7" }}>
            <td style={{ padding: 4, border: "1px solid #ccc" }}>
              <input
                type="date"
                name="Date"
                value={entry.Date}
                onChange={handleEntryChange}
                style={{
                  width: "100%",
                  padding: 6,
                  borderRadius: 3,
                  border: "1px solid #ccc",
                }}
              />
            </td>
            <td style={{ padding: 4, border: "1px solid #ccc" }}>
              <input
                type="text"
                name="Description"
                value={entry.Description}
                onChange={handleEntryChange}
                placeholder="Description"
                style={{
                  width: "100%",
                  padding: 6,
                  borderRadius: 3,
                  border: "1px solid #ccc",
                }}
              />
            </td>
            <td style={{ padding: 4, border: "1px solid #ccc" }}>
              <input
                type="number"
                name="Amount"
                value={entry.Amount}
                onChange={handleEntryChange}
                placeholder="Amount"
                step="0.01"
                style={{
                  width: "100%",
                  padding: 6,
                  borderRadius: 3,
                  border: "1px solid #ccc",
                  textAlign: "right",
                }}
              />
            </td>
            <td style={{ padding: 4, border: "1px solid #ccc" }}>
              <select
                name="Category"
                value={entry.Category}
                onChange={handleEntryChange}
                style={{
                  width: "100%",
                  padding: 6,
                  borderRadius: 3,
                  border: "1px solid #ccc",
                }}
              >
                {categoryOptions.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
            </td>
            <td
              style={{
                padding: 4,
                border: "1px solid #ccc",
                textAlign: "center",
              }}
            >
              <button
                onClick={addManualEntry}
                style={{
                  padding: "6px 12px",
                  backgroundColor: "#0078d7",
                  color: "white",
                  border: "none",
                  borderRadius: 3,
                  cursor: "pointer",
                }}
              >
                ➕
              </button>
            </td>
          </tr>
        </tbody>
      </table>
      <input
        type="file"
        accept=".csv"
        onChange={handleFileUpload}
        style={{ marginBottom: 20 }}
      />
      {loading && <p>Processing… ⏳</p>}

      {/* Entries Table & Charts Side-by-Side */}
      {entries.length > 0 && (
        <div style={{ display: "flex", gap: 20 }}>
          {/* Table */}
          <div style={{ flex: 1, overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                marginBottom: 0,
                tableLayout: "fixed",
              }}
            >
              <thead>
                <tr style={{ backgroundColor: "#0078d7", color: "white" }}>
                  <th style={{ padding: 8 }}>Date</th>
                  <th style={{ padding: 8 }}>Description</th>
                  <th style={{ padding: 8, textAlign: "right" }}>Amount</th>
                  <th style={{ padding: 8 }}>Category</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((r, i) => (
                  <tr
                    key={i}
                    style={{
                      backgroundColor: i % 2 === 0 ? "#f0f8ff" : "white",
                    }}
                  >
                    <td style={{ padding: 8 }}>{r[0]}</td>
                    <td style={{ padding: 8 }}>{r[1]}</td>
                    <td
                      style={{
                        padding: 8,
                        textAlign: "right",
                        fontWeight: "bold",
                      }}
                    >
                      {format(Number(r[2]))}
                    </td>
                    <td
                      style={{
                        padding: 8,
                        backgroundColor: getCategoryColor(r[3]),
                      }}
                    >
                      {r[3]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Charts */}
          {data && (
            <div style={{ flex: 1 }}>
              <div style={{ marginBottom: 20 }}>
                {data.category_summary.map((c) => (
                  <div
                    key={c.Category}
                    style={{ whiteSpace: "pre-wrap", marginBottom: 8 }}
                  >
                    {`🗂️ ${c.Category}\nTransactions: ${
                      c.TransactionCount
                    }\nTotal: ${format(c.TotalAmount)}\nWithdrawals: ${format(
                      c.Withdrawals
                    )}\nDeposits: ${format(c.Deposits)}`}
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", gap: 20 }}>
                <div style={{ flex: 1, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={data.category_summary}
                        dataKey="TotalAmount"
                        nameKey="Category"
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        label
                      >
                        {data.category_summary.map((e, idx) => (
                          <Cell key={idx} fill={getPieColor(e.Category)} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ flex: 1, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={data.category_summary}
                      margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
                    >
                      <XAxis dataKey="Category" />
                      <YAxis />
                      <Tooltip formatter={(v: number) => format(v)} />
                      <Legend />
                      <Bar dataKey="Deposits" name="Deposits" barSize={20} />
                      <Bar
                        dataKey="Withdrawals"
                        name="Withdrawals"
                        barSize={20}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
