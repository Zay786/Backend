const express = require("express");
const cors = require("cors");
const bcrypt = require("bcrypt");
const axios = require("axios");
const knexLib = require("knex");

require("dotenv").config();

const createDbConnection = () => {
  if (process.env.DATABASE_URL) {
    return {
      connectionString: process.env.DATABASE_URL,
      ssl: { rejectUnauthorized: false },
    };
  }

  return {
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    port: process.env.DB_PORT,
    ssl: { rejectUnauthorized: false },
  };
};

const knex = knexLib({
  client: "pg",
  connection: createDbConnection(),
  searchPath: ["public"],
});

const app = express();

const registerApiRoute = (method, path, handler) => {
  app[method](path, handler);
  app[method](`/api${path}`, handler);
};

const getMlApiBaseUrl = (req) => {
  if (process.env.ML_API_URL) {
    return process.env.ML_API_URL.replace(/\/$/, "");
  }

  const protocol = req.headers["x-forwarded-proto"] || "https";
  return `${protocol}://${req.headers.host}/api/ml`;
};

app.use(cors({ origin: "*" }));
app.use(express.json());

registerApiRoute("get", "/health", async (_req, res) => {
  try {
    await knex.raw("SELECT 1");
    return res.json({ ok: true });
  } catch (error) {
    console.error("HEALTH CHECK ERROR:", error.message);
    return res.status(500).json({ ok: false, error: "Database unavailable" });
  }
});

registerApiRoute("post", "/login", async (req, res) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      return res.status(400).json({ error: "Missing credentials" });
    }

    const user = await knex("logistics_users").where("username", username).first();

    if (!user) {
      return res.status(401).json({ error: "Invalid username" });
    }

    const valid = await bcrypt.compare(password, user.password_hash);
    if (!valid) {
      return res.status(401).json({ error: "Invalid password" });
    }

    return res.json({
      message: "Login successful",
      userId: user.user_id,
    });
  } catch (error) {
    console.error("LOGIN ERROR:", error.message);
    return res.status(500).json({ error: "Server error" });
  }
});

registerApiRoute("post", "/quotation", async (req, res) => {
  try {
    const data = req.body;

    const response = await axios.post(
      `${getMlApiBaseUrl(req)}/generate`,
      data,
      { timeout: 10000 }
    );

    const { price, pdf_url, pdf_base64, pdf_file_name } = response.data;

    await knex("quotations").insert({
      customer_name: data.name,
      company_name: data.company,
      email: data.email,
      origin: data.origin,
      destination: data.destination,
      commodity: data.commodity,
      weight_tons: data.weight,
      service_type: data.service,
      predicted_price: price,
    });

    return res.json({
      success: true,
      price,
      pdf_url,
      pdf_base64,
      pdf_file_name,
    });
  } catch (error) {
    console.error("QUOTATION ERROR:", error.message);

    if (error.response) {
      console.error("ML API ERROR:", error.response.data);
    }

    return res.status(500).json({ error: "Quotation failed" });
  }
});

module.exports = app;
