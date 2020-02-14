package com.nonint.nlp.transformers;

import com.google.common.collect.ImmutableList;
import com.google.common.io.Resources;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;

@RunWith(JUnit4.class)
public class GPT2TokenizerTest {
    final String TOKENIZE_TEST_FILE = "gpt2_tokenize_test.json";
    final String ENCODE_TEST_FILE = "gpt2_encode_test.json";

    GPT2Tokenizer tokenizer;

    @Before
    public void init() throws IOException, ParseException {
        tokenizer = GPT2Tokenizer.fromPretrained("gpt2");
    }

    @Test
    public void testTokenize() throws IOException, ParseException {
        JSONObject root = (JSONObject)new JSONParser().parse(Resources.toString(Resources.getResource(TOKENIZE_TEST_FILE), Charset.defaultCharset()));
        for(Object okey : root.keySet()) {
            String originalSentence = okey.toString();
            ImmutableList<String> ourTokens = tokenizer.tokenize(originalSentence);

            List<String> tokens = Arrays.stream(((JSONArray)root.get(okey)).toArray()).map(o -> o.toString()).collect(Collectors.toList());
            assertThat(ourTokens).containsExactlyElementsOf(tokens);
        }
    }

    @Test
    public void testEncode() throws IOException, ParseException {
        JSONObject root = (JSONObject)new JSONParser().parse(Resources.toString(Resources.getResource(ENCODE_TEST_FILE), Charset.defaultCharset()));
        for(Object okey : root.keySet()) {
            String originalSentence = okey.toString();
            Tokenizer.EncoderResult result = tokenizer.encode(originalSentence, 128);
            JSONObject encMap = (JSONObject) root.get(okey);

            List<Integer> inputIds = Arrays.stream(((JSONArray)encMap.get("input_ids")).toArray()).map(o -> Integer.parseInt(o.toString())).collect(Collectors.toList());
            assertThat(result.inputIds).containsExactlyElementsOf(inputIds);
            List<Integer> attMask = Arrays.stream(((JSONArray)encMap.get("attention_mask")).toArray()).map(o -> Integer.parseInt(o.toString())).collect(Collectors.toList());
            assertThat(result.attentionMask).containsExactlyElementsOf(attMask);
            if(encMap.containsKey("num_truncated_tokens")) {
                assertThat(Integer.parseInt(encMap.get("num_truncated_tokens").toString())).isEqualTo(result.truncatedTokensCount);
            }
        }
    }
}
