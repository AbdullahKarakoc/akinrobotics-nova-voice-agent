-- Sample operational data for the fictional company "Nova Robotik".
-- Loaded into SQLite on first start (see app/tools/database.py).

CREATE TABLE products (
    code            TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    price_try       INTEGER NOT NULL,   -- list price in Turkish lira, VAT excluded
    stock           INTEGER NOT NULL,
    lead_time_weeks INTEGER NOT NULL,   -- delivery time when out of stock
    aliases         TEXT NOT NULL       -- comma-separated words users may say instead of the name
);

INSERT INTO products VALUES
    ('R2', 'Rehber R2',   1450000, 3, 6,  'rehber,guide,reception,karşılama,humanoid,insansı'),
    ('T5', 'Taşıyıcı T5',  890000, 0, 5,  'taşıyıcı,carrier,logistics,lojistik,transport,taşıma'),
    ('C3', 'Temizlik C3',  520000, 7, 4,  'temizlik,cleaning,cleaner,temizleme'),
    ('E1', 'Eğitim E1',     18500, 42, 2, 'eğitim,education,educational,kit');

CREATE TABLE orders (
    order_no     INTEGER PRIMARY KEY,
    customer     TEXT NOT NULL,
    product_code TEXT NOT NULL REFERENCES products(code),
    quantity     INTEGER NOT NULL,
    status       TEXT NOT NULL,         -- preparing | in_production | shipped | delivered | cancelled
    order_date   TEXT NOT NULL,
    eta          TEXT                   -- estimated delivery date (NULL when delivered/cancelled)
);

-- Order numbers have 3 digits that are short to say ("üç yüz kırk"), so speech recognition gets them right.
INSERT INTO orders VALUES
    (120, 'Anadolu AVM',             'R2', 2,  'delivered',     '2026-08-04', NULL),
    (250, 'Kuzey Lojistik',          'T5', 4,  'in_production', '2026-09-15', '2026-10-27'),
    (340, 'Ege Üniversite Hastanesi','C3', 1,  'shipped',       '2026-10-01', '2026-10-12'),
    (460, 'Bilim Koleji',            'E1', 20, 'preparing',     '2026-10-07', '2026-10-16'),
    (580, 'Başkent Müzesi',          'R2', 1,  'cancelled',     '2026-09-20', NULL);

CREATE TABLE service_centers (
    city    TEXT PRIMARY KEY,
    address TEXT NOT NULL,
    phone   TEXT NOT NULL
);

INSERT INTO service_centers VALUES
    ('Eskişehir', 'Organize Sanayi Bölgesi 12. Cadde No: 4, Odunpazarı', '0222 555 10 10'),
    ('İstanbul',  'İkitelli OSB, Mermerciler Sitesi No: 21, Başakşehir',  '0212 555 20 20'),
    ('Ankara',    'Ostim OSB, 1234. Sokak No: 8, Yenimahalle',           '0312 555 30 30'),
    ('İzmir',     'Atatürk OSB, 10003 Sokak No: 15, Çiğli',              '0232 555 40 40'),
    ('Antalya',   'Antalya OSB, 3. Kısım, Döşemealtı',                    '0242 555 50 50');
