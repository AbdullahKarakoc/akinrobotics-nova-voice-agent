-- Sample after-sales data for the fictional company "Nova Robotik": delivered robots,
-- their remote telemetry, error codes and service tickets. Loaded into an in-memory
-- SQLite database when the MCP server starts (see server.py).

CREATE TABLE models (
    code            TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    warranty_months INTEGER NOT NULL      -- standard warranty, from the delivery date
);

INSERT INTO models VALUES
    ('R2', 'Rehber R2',   24),
    ('T5', 'Taşıyıcı T5', 24),
    ('C3', 'Temizlik C3', 24),
    ('E1', 'Eğitim E1',   12);

CREATE TABLE robots (
    serial           TEXT PRIMARY KEY,    -- model code + 4 unique digits (round hundreds, short to say,
                                          -- never a year); customers usually say the digits
    model_code       TEXT NOT NULL REFERENCES models(code),
    delivered        TEXT NOT NULL,
    extended_until   TEXT,                -- extended warranty end (NULL = standard warranty only)
    last_maintenance TEXT,                -- NULL = no preventive maintenance needed (education kit)
    firmware         TEXT NOT NULL,
    online           INTEGER NOT NULL,
    battery_pct      INTEGER NOT NULL,
    last_seen        TEXT NOT NULL,
    error_code       TEXT                 -- active error, NULL = none
);

INSERT INTO robots VALUES
    ('R2-3100', 'R2', '2026-08-04', NULL,         '2026-08-04', '3.4.1',  1, 82, '2026-10-11 09:42', NULL),
    ('R2-3200', 'R2', '2026-08-04', NULL,         '2026-08-04', '3.4.1',  0,  4, '2026-10-10 22:15', 'E-101'),
    ('T5-4500', 'T5', '2024-05-20', NULL,         '2025-11-18', '2.9.0',  1, 64, '2026-10-11 09:40', 'E-204'),
    ('C3-6800', 'C3', '2025-03-10', NULL,         '2026-09-02', '1.12.3', 1, 97, '2026-10-11 09:41', NULL),
    ('R2-2800', 'R2', '2023-11-15', '2028-11-15', '2026-05-12', '3.3.0',  1, 55, '2026-10-11 09:38', 'E-310'),
    ('E1-7200', 'E1', '2025-09-01', NULL,         NULL,         '1.0.7',  1, 71, '2026-10-09 15:20', NULL);

CREATE TABLE error_codes (
    code      TEXT PRIMARY KEY,
    issue_tr  TEXT NOT NULL,
    issue_en  TEXT NOT NULL,
    advice_tr TEXT NOT NULL,
    advice_en TEXT NOT NULL
);

INSERT INTO error_codes VALUES
    ('E-101', 'Batarya kritik seviyede, robot şarj istasyonuna dönemedi.',
              'Battery critically low; the robot could not return to its charging dock.',
              'Robotu elle şarj istasyonuna yerleştirin. Sorun tekrarlarsa servis talebi açın.',
              'Place the robot on its charging dock by hand. If it happens again, open a service request.'),
    ('E-204', 'LiDAR sensör camı kirli, robot yavaş ilerliyor.',
              'The LiDAR sensor window is dirty, so the robot moves slowly.',
              'Sensör camını kuru ve yumuşak bir bezle silin.',
              'Wipe the sensor window with a dry, soft cloth.'),
    ('E-310', 'Son yazılım güncellemesi yüklenemedi.',
              'The latest software update could not be installed.',
              'Robotu bu gece şarjda bırakın, güncelleme otomatik olarak tekrar denenecek.',
              'Leave the robot charging tonight; the update will be retried automatically.');

CREATE TABLE tickets (
    ticket_no           INTEGER PRIMARY KEY,
    serial              TEXT NOT NULL REFERENCES robots(serial),
    opened              TEXT NOT NULL,
    status              TEXT NOT NULL,    -- open | technician_assigned | waiting_parts | resolved
    issue_tr            TEXT NOT NULL,
    issue_en            TEXT NOT NULL,
    technician_visit    TEXT,
    expected_resolution TEXT,
    resolution_tr       TEXT,
    resolution_en       TEXT
);

-- Ticket numbers have 2 digits ("yirmi altı").
INSERT INTO tickets VALUES
    (15, 'T5-4500', '2026-09-28', 'resolved',            'Sol tahrik tekerleğinden ses geliyor.', 'Noise from the left drive wheel.',
         '2026-09-30', NULL,         'Tekerlek rulmanı değiştirildi.', 'The wheel bearing was replaced.'),
    (26, 'R2-3200', '2026-10-10', 'technician_assigned', 'Robot şarj istasyonuna dönmüyor.',      'The robot does not return to its charging dock.',
         '2026-10-12', '2026-10-12', NULL, NULL),
    (37, 'R2-2800', '2026-10-06', 'waiting_parts',       'Dokunmatik ekran yanıt vermiyor.',      'The touch screen does not respond.',
         NULL,         '2026-10-20', NULL, NULL),
    (48, 'C3-6800', '2026-10-09', 'open',                'Fırça motoru ara ara duruyor.',         'The brush motor stops now and then.',
         NULL,         NULL,         NULL, NULL);
