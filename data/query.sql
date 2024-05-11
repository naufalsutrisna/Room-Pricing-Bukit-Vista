SELECT 
    u.unit_name, u.bedroom, u.bathroom,
    u.beds, u.guests, ut.name AS 'type',
    ua.*,
    p.property_name, p.property_bedrooms,
    p.lat, p.lng, p.distance_to_coastline,
    br.room_name, b.booking_received_timestamp,
    b.booking_check_in, b.booking_check_out,
    DATEDIFF(b.booking_check_in, b.booking_received_timestamp) AS booking_window,
    DATEDIFF(b.booking_check_out, b.booking_check_in) AS stay_duration_in_days,
    r.review_sentiment_score, r.rating,
    b.booking_earned, 
    b.booking_earned / DATEDIFF(b.booking_check_out, b.booking_check_in) AS average_daily_rate
FROM unit u
JOIN unit_type ut ON u.unit_type_id = ut.unit_type_id
JOIN property p ON p.property_id = u.property_id
JOIN beds24_room_to_unit brtu ON brtu.unit_id = u.unit_id
JOIN beds24_room br ON br.room_id = brtu.room_id
JOIN booking b ON b.property_id = u.property_id
JOIN booking_to_units bto ON bto.booking_id = b.booking_id
JOIN unit_amenities ua ON ua.unit_id = u.unit_id
JOIN reviews r ON r.booking_id = b.booking_id
WHERE 
        b.booking_check_in >= '2023-01-01' AND 
        b.booking_check_in <= '2023-12-31'