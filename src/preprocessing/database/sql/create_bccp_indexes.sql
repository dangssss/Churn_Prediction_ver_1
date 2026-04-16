DO $$
DECLARE
  r RECORD;
  create_stmt text;
BEGIN
  FOR r IN SELECT tablename FROM pg_tables 
  WHERE schemaname = 'public' AND tablename LIKE 'bccp_orderitem_%' AND tablename !~ 'bccp_orderitem_[0-9]{4}$'
  LOOP
    create_stmt := 'CREATE INDEX IF NOT EXISTS idx_' || r.tablename || '_code_time 
                    ON public.' || r.tablename || '(cms_code_enc, sending_time)';
    EXECUTE create_stmt;
    RAISE NOTICE 'Created index on %.%', 'public', r.tablename;
  END LOOP;
END $$;

ANALYZE public.cas_customer;
ANALYZE public.cas_info;
ANALYZE public.cms_complaint;
