import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from closetscan.mcp_server import Wardrobe, handle
from closetscan.catalogue import load_catalogue
ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / 'tests/fixtures/catalogue'
class ReleaseTests(unittest.TestCase):
    def test_catalogue_assets(self):
        garments = load_catalogue(str(CATALOGUE))
        self.assertGreater(len(garments), 0)
        for g in garments:
            for v in g.views:
                self.assertTrue((CATALOGUE/v.file).is_file(), v.file)
    def test_protocol(self):
        messages = ['{', '[]', 'null'] + [json.dumps(m) for m in [
            {'jsonrpc':'2.0','method':'notifications/initialized'},
            {'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'unsupported'}},
            {'jsonrpc':'2.0','id':2,'method':'tools/list'},
            {'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'list_garments','arguments':{'limit':2}}}]]
        p = subprocess.run([sys.executable,'-m','closetscan.mcp_server',str(CATALOGUE)],input='\n'.join(messages)+'\n',text=True,capture_output=True,check=True)
        replies = [json.loads(line) for line in p.stdout.splitlines()]
        self.assertEqual(len(replies),6)
        self.assertEqual([r['error']['code'] for r in replies[:3]],[-32700,-32600,-32600])
        self.assertEqual(replies[3]['result']['protocolVersion'],'2024-11-05')
        self.assertEqual(len(replies[4]['result']['tools']),7)
        self.assertEqual(json.loads(replies[5]['result']['content'][0]['text'])['returned'],2)
    def test_notification_cannot_write(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(handle({'jsonrpc':'2.0','method':'tools/call','params':{'name':'log_wear','arguments':{'garment_id':'g000'}}},Wardrobe(d)))
            self.assertEqual(list(Path(d).iterdir()),[])
    def test_wear_is_separate(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            for name in ('manifest.json','dedup.json','attributes.json'):
                shutil.copy(CATALOGUE/name,d)
            before=(Path(d)/'manifest.json').read_bytes()
            w=Wardrobe(d)
            def call(name,args):
                return handle({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':name,'arguments':args}},w)['result']
            gid=w.garments[0].id
            self.assertTrue(call('log_wear',{'garment_id':gid,'date':'invalid'})['isError'])
            self.assertFalse(call('log_wear',{'garment_id':gid,'date':'2026-09-13'})['isError'])
            self.assertEqual(json.loads(call('get_wear_history',{})['content'][0]['text'])['total_entries'],1)
            self.assertEqual((Path(d)/'manifest.json').read_bytes(),before)
    def test_html_script_escape(self):
        try:
            from closetscan.html_export import render
        except ImportError:
            self.skipTest('needs .[pipeline]')
        data={'meta':{'n_garments':1,'n_rows':1,'clip':'</script><script>alert(1)</script>','span':'1s','has_plates':True},'items':[]}
        output=render(data,'<bad>')
        self.assertNotIn('</script><script>alert(1)',output)
        self.assertIn('&lt;bad&gt;',output)
    def test_notebook(self):
        nb=json.loads((ROOT/'closetscan_colab.ipynb').read_text())
        for c in nb['cells']:
            if c['cell_type']=='code':
                compile(c['source'],'<notebook>','exec')
                self.assertEqual(c['outputs'],[])
if __name__=='__main__': unittest.main()
